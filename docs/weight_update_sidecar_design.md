# Model and Trie update sidecar design

## Purpose and scope

A sidecar manages resource updates for the GR service instances in one Pod.
Checkpoint updates change model weights only; Trie updates change constraint
contents only. Model architecture, tokenizer, resource formats, and serving
configuration remain unchanged.

This document describes the architecture, responsibilities, the agreed FileSystem
interface, and a proposed user configuration. Other APIs, configuration validation
rules, and implementation details are deferred.

## Architecture

<img src="assets/weight_update_sidecar_flow.svg" alt="Sidecar tasks, update modes and update processes" width="650">

One Sidecar Process owns multiple Tasks. Each Task independently discovers,
downloads, and validates a resource version, then applies it to its target service
instances. The diagram shows a checkpoint Task and a Trie Task.

`Node` means one GR service instance, not a physical host. A Pod commonly has
eight single-GPU instances, but the target instance count is configurable. An
instance can own multiple workers; its resource update must cover every worker.

## Core abstractions

| Abstraction | Responsibility |
| --- | --- |
| **Sidecar** | Run and monitor Tasks, and coordinate updates that target the same instances. |
| **Task** | Own one resource's source, polling interval, local storage, validation rules, target instances, update configuration, version state, and progress. |
| **FileSystem** | Discover published versions and download a selected version; hide storage-specific behavior. |
| **Update Mode** | Decide how updates are scheduled across the target instances. |
| **Update Process** | Define the steps performed on each selected instance. |

Update Mode and Update Process are independent: the mode controls **which
instances update together**, while the process controls **what happens on each
instance**. Neither is hard-coded into the resource type. A Task must choose a
combination supported by its resource and the server.

### Task

The checkpoint Task and Trie Task have separate sources, configurations, and
version state. A resource update in one Task does not imply an update in the
other. Each Task selects its FileSystem, Update Mode, and Update Process.

Discovery and download can proceed independently across Tasks. Update execution
on overlapping service instances must be coordinated so checkpoint and Trie
reloads cannot race with each other.

### FileSystem

Use pluggable implementations such as HDFS, P2P, and other storage systems. Start
with HDFSDownloader; adding another implementation does not change Task scheduling
or instance update behavior.

The initial interface is:

```python
from pathlib import Path
from typing import Protocol


class FileSystem(Protocol):
    async def get_latest_version(self, address: str) -> str | None: ...

    async def download(
        self, address: str, version: str, target_dir: Path
    ) -> None: ...
```

- `get_latest_version` takes the monitored address and returns the latest published
  version, or `None` if no version is available. Storage access failures raise an
  exception rather than returning `None`.
- `download` takes the same monitored address, a selected version, and a local
  target directory. HDFSDownloader resolves the version's storage location and
  downloads that exact version. If it is unavailable, raise an exception rather
  than silently downloading a newer version.

A Task compares the discovered version with its completed version before download:

```python
version = await filesystem.get_latest_version(address)
if version is not None and version != completed_version:
    await filesystem.download(address, version, target_dir)
    # Validate the downloaded resource, then execute the configured update.
```

FileSystem owns version discovery and download. Validation, instance scheduling,
and completed-version state belong to the Task/controller. Downloads remain
pinned to the selected immutable version even if another version is published.

Download once into Pod-shared storage, then validate completeness, checksums, and
resource compatibility. Publish a usable local resource only after validation.
Instances continue serving throughout discovery, download, and validation.

### Update Mode

- **Rolling Update:** update instances in groups of at most `n`, defaulting to 1.
  Wait for the current group to finish before starting the next. Instances outside
  the group continue serving.
- **Sync Update:** dispatch updates to all target instances concurrently, without
  rolling groups. This means concurrent execution, not an atomic version switch
  at exactly the same instant.

Both modes execute the Task's selected Update Process on each instance and track
completion or failure separately for every target.

### Update Process

- **PausedUpdate Process:** pause admission and drain existing requests, reload
  the resource, resume scheduling, and run warmup. Business admission remains
  closed until warmup succeeds. The sequence is **Pause/Drain -> Reload ->
  Resume/Warmup**.
- **OnlineUpdate Process:** directly reload the resource while serving continues.
  The server prepares the new resource, safely publishes it for new requests,
  and retains the old resource until existing requests no longer reference it.
  The sidecar does not issue pause, drain, resume, or warmup steps separately.

Each process owns its instance-level completion and failure conditions. The
sidecar delegates resource loading and execution-resource maintenance to the
server; it does not manipulate model or Trie objects itself.

## Initial Task combinations

| Task | Update Mode | Update Process |
| --- | --- | --- |
| **Task_ckpt** | Rolling Update, at most `n` instances per group | PausedUpdate Process |
| **Task_Trie** | Sync Update, all target instances concurrently | OnlineUpdate Process |

### Checkpoint Task

After preparing a valid checkpoint, roll through the instances using PausedUpdate
Process. Reload internally loads weights on every worker, refreshes derived weight
caches, and preserves or rebuilds affected Graph resources. Warmup verifies that
the updated instance can execute its workload before business traffic resumes.

### Trie Task

After preparing a valid Trie artifact, invoke OnlineUpdate Process on all target
instances concurrently. The resource manager creates a new immutable
ConstraintTable and device Backend, then replaces the published resource instead
of modifying the active table in place.

Online Trie reload requires server-side version ownership: each request keeps one
Trie version for its full lifetime. New requests use the new version after it is
ready; existing requests finish with the old one. Retain matching Backend and
Graph resources until their users finish, then release them. Preparation failure
leaves the current resource active.

This online switching capability must be implemented and verified; creating a
new Backend alone does not make reload safe without pause/drain. Any resource
preparation needed for online execution belongs inside the server reload process.

## Shared update lifecycle

Every Task follows **Check update -> Download -> Check valid -> Apply update**.
The Update Mode expands Apply update across instances, and the Update Process
executes the per-instance actions through HTTP control requests. These control
requests are independent of the inference protocol.

Track the target version and each instance's progress separately. A Task marks a
version complete only after all target instances finish their selected process.
Updates may temporarily serve different versions across instances. Finish or
recover the current version before applying another version of the same Task.

A failed download or validation does not affect serving. A failed rolling update
stops subsequent groups and retains the failed instance's update slot until
recovery. An online update failure retains the last usable resource whenever it
has not been replaced. Resume from recorded and reconciled per-instance state
after a retry or sidecar restart; do not assume a timed-out call did not execute.

## User configuration example

The user provides a Python configuration file exporting a `config: SidecarConfig`
object. The framework loads that object and manages startup and execution; the
configuration file does not construct a running Sidecar or call `run()`.

FileSystem, Update Mode, and Update Process implementations are passed as Python
objects, so users can supply custom implementations directly. The example shows
object composition; constructor details remain subject to the API discussion.

```python
from pathlib import Path


config = SidecarConfig(
    state_dir=Path("/updates/state"),
    instances=[
        Instance(id=f"node-{i}", control_url=f"http://127.0.0.1:{9100 + i}")
        for i in range(8)
    ],
    tasks=[
        UpdateTask(
            name="ckpt",
            resource_type="checkpoint",
            filesystem=HDFSDownloader(),
            address="hdfs://cluster/models/gr/checkpoints",
            target_dir=Path("/updates/checkpoints"),
            poll_interval_seconds=30,
            update_mode=RollingUpdate(max_concurrent_updates=1),
            update_process=PausedUpdateProcess(),
        ),
        UpdateTask(
            name="trie",
            resource_type="trie",
            filesystem=HDFSDownloader(),
            address="hdfs://cluster/models/gr/tries",
            target_dir=Path("/updates/tries"),
            poll_interval_seconds=60,
            update_mode=SyncUpdate(),
            update_process=OnlineUpdateProcess(),
        ),
    ],
)
```

- `instances` lists service control endpoints; its length is independent of the
  physical GPU count. Tasks can target `all` or a list of instance IDs.
- `filesystem` is an implementation object. The Task passes `address` to both
  version lookup and download, together with the selected version for download.
- `target_dir` is the Task's shared local cache root. Download into a temporary
  version directory below it and publish only after validation.
- `update_mode` controls instance scheduling; `update_process` controls each
  instance's actions. The initial combinations are checkpoint + RollingUpdate +
  PausedUpdateProcess and Trie + SyncUpdate + OnlineUpdateProcess.
- `state_dir` stores independent version and per-instance progress for each Task.
  Updates targeting the same instance are coordinated across Tasks, even when
  discovery and download loops run independently.

Resource validation/reload selection and process-specific settings will be refined
in the remaining API discussion. No additional interfaces are specified here.

## Validation goals

- Tasks keep independent configuration, versions, and progress while coordinating
  update execution on shared target instances.
- FileSystem implementations download the selected immutable version even when
  a newer version is published during download.
- Rolling Update respects `n`; Sync Update dispatches to all configured targets.
- PausedUpdate preserves drain/reload/resume/warmup ordering and admission state.
- Online Trie reload preserves in-flight request versions and Graph references,
  releases old resources safely, and retains service on preparation failure.
- Both resource types support observable failure, retry, and restart recovery;
  successful updates produce the expected new weights or Trie candidates.
