## Environment variables</p>


UPC++ DepSpawn supports several environtment variables that control its behavior.

### Inherited from DepSpawn


 
  - `DEPSPAWN_NUM_THREADS`: number of threads per process.
  
  - `DEPSPAWN_TASK_QUEUE_LIMIT`: only meaningful when the runtime has been compiled with `DEPSPAWN_FAST_START` and the Exact Match Mode (EMM) has not been requested. The master thread runs ready tasks when
  	- the number of ready tasks exceeds this value, or 
  	- there more ready tasks than threads and the number of local not yet ready tasks exceeds this value.

### Exclusive to UPC++ DepSpawn

  - `UPCXX_DEPSPAWN_ADV_RATE`: Each how many spawns does the main thread invoke `upcxx::progress()`.
  - `UPCXX_DEPSPAWN_CLR_RATE`: Each how many spawns does the main thread try to:
  	-  clear the TDG of finished tasks, and
  	-  perform synchronizations related to dependencies, if needed
  - `UPCXX_DEPSPAWN_MIN_REPORT`: Minimum progress in the TDG that is worth reporting to other processes, measured in number of tasks.
  - `UPCXX_DEPSPAWN_STOP_RATE`: Number of live tasks in the local TDG that, when reached, leads the main thread to stop spawning new tasks. In this situation the thread runs ready tasks and advances the runtime.
  - `UPCXX_DEPSPAWN_MIN_LOCAL`: Minimum numer of tasks among the  `UPCXX_DEPSPAWN_STOP_RATE` tasks that lead the main thread to stop spawning new tasks that must be local.
  - `UPCXX_DEPSPAWN_MAX_CACHE`: Maximum cache size, in items
  - `UPCXX_DEPSPAWN_SLACK_CACHE`: Slack temporarily allowed above the maximum cache size before the cache is trimmed.
  - `UPCXX_DEPSPAWN_EXACT_MATCH`: Request Exact Match Mode (EMM), in which the runtime does not check for overlaps of arguments to detect the dependencies, but only on coincidence of starting address and owner process.
  - `UPCXX_DEPSPAWN_YIELD`: When defined, threads that are waiting for a cache item to be received from its owner try to advance the runtime during the wait.
  - `UPCXX_DEPSPAWN_FWD_NOTICE`: When defined, the source of a dependency notifies the processes it knows that have dependent tasks as soon as the dependency is fulfilled.
  - `UPCXX_DEPSPAWN_PREFETCH`: When defined, when a task is readied, its read-only arguments are prefetched to the cache. Activates `UPCXX_DEPSPAWN_YIELD`.
  - `UPCXX_DEPSPAWN_ACTIVE_WAIT`: When defined, idle threads try to advance the runtime.

  
 
