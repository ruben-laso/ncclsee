#include <stdio.h>
#include <pthread.h>
#include <string.h>
#include <linux/limits.h>
#include <sys/time.h>
#include <sys/types.h>
#include <sys/syscall.h>
#include <unistd.h>
#include <inttypes.h>
#include <x86intrin.h>
#include <stdatomic.h>
#include "profiler.h"


#define __hidden __attribute__ ((visibility("hidden")))
#define GROUP_POOL_SIZE 128
#define COLL_POOL_SIZE 128
#define PROXY_POOL_SIZE 128


static const char plugin_name[32] = "A Fools Hope";
static const int defaultEActivationMask = ncclProfileColl | ncclProfileP2p | ncclProfileProxyOp;

enum nccl_colls {
    nccl_allreduce,
    nccl_broadcast,
    nccl_reduce,
    nccl_reduce_scatter,
    nccl_allgather,
    nccl_alltoall,
    nccl_unknown, // For unexpected cases
    nccl_num_colls // Keeps track of total primitives
};

static const char* nccl_coll_names[nccl_num_colls] = {
    "AllReduce",
    "Broadcast",
    "Reduce",
    "ReduceScatter",
    "AllGather",
    "AllToAll",
    "Unknown"
};

/* static const int groupPoolSize = 128; */
/* static const int collPoolSize = 128; */

// Counters for each collective type.
// static uint64_t allReduceCount = 0;
// static uint64_t broadcastCount = 0;
// static uint64_t reduceCount = 0;
// static uint64_t reduceScatterCount = 0;
// static uint64_t allGatherCount = 0;
// static uint64_t allToAllCount = 0;
// static uint64_t unknownCount = 0;
// static uint64_t allReduceBytes = 0;
// static uint64_t broadcastBytes = 0;
// static uint64_t reduceBytes = 0;
// static uint64_t reduceScatterBytes = 0;
// static uint64_t allGatherBytes = 0;
// static uint64_t allToAllBytes = 0;
// static uint64_t unknownBytes = 0;
// static double allReduceTime = 0;
// static double broadcastTime = 0;
// static double reduceTime = 0;
// static double reduceScatterTime = 0;
// static double allGatherTime = 0;
// static double allToAllTime = 0;
// static double unknownTime = 0;

static struct {
    uint64_t count;
    uint64_t bytes;
    double time;
} stats[nccl_num_colls] = {};

struct group;
struct context;
struct proxyOp;

struct proxyOp {
  uint8_t type;           // ncclProfileProxyOp
  struct collective* parent; // The collective that spawned this proxy op
  pid_t pid;
};


struct group {
  uint8_t type;
  struct context* ctx;
  int refCount;
  double startTs;
  double stopTs;
};

struct collective {
  uint8_t type;
  struct group* parent;
  int refCount;
  double startTs;
  double stopTs;
};



struct context {
 int groupIndex;
 struct group groupPool[GROUP_POOL_SIZE];
 int collIndex;
 struct collective collPool[COLL_POOL_SIZE];
 int proxyIndex;
 struct proxyOp proxyPool[PROXY_POOL_SIZE];
};

static int initialized;             // initialization counter for profiler
/* static FILE *debug_file; */
static pthread_mutex_t lock = PTHREAD_MUTEX_INITIALIZER;
static pid_t pid;
static double startTime;


static double freq = -1;
__hidden void calibrate() {
  struct timeval tv;
  gettimeofday(&tv, NULL);
  uint64_t timeCycles = __rdtsc();
  double time = - tv.tv_sec*1e6 - tv.tv_usec;
  uint64_t total = 0ULL;
  for (int i = 0; i < 10000; i++) total += __rdtsc();
  gettimeofday(&tv, NULL);
  timeCycles = __rdtsc() - timeCycles;
  time += tv.tv_sec*1e6 + tv.tv_usec;
  freq = timeCycles / time;
}

__hidden double gettime(void) {
  return __rdtsc() / freq;
}


__hidden ncclResult_t Profiler_Init(void** context, int* eActivationMask) {

  pthread_mutex_lock(&lock);
  if (__atomic_fetch_add(&initialized, 1, __ATOMIC_RELAXED) == 0) {
    // first thread initializes event mask, environment and detach pool
    const char* str;
    str = getenv("NCCL_PROFILE_EVENT_MASK");
    __atomic_store_n(eActivationMask, str ? atoi(str) : defaultEActivationMask, __ATOMIC_RELAXED);
    pid = getpid();
    calibrate();
    startTime = gettime();

  }
  pthread_mutex_unlock(&lock);

  //fprintf(stderr, "Profiler_Init: eActivationMask = %d\n", *eActivationMask);
    // Allocate memory for the context
  struct context* ctx = (struct context*)calloc(1, sizeof(struct context));
  if (ctx == NULL) {
      fprintf(stderr, "Profiler_Init: Failed to allocate memory for context\n");
      return ncclInternalError; // Return an appropriate NCCL error code
  }
  ctx->groupIndex = 0;
  ctx->collIndex = 0;
  ctx->proxyIndex = 0;
  // Assign the context to the output parameter
  *context = ctx;


  return ncclSuccess;
}

__hidden ncclResult_t Profiler_Finalize(void* context) {

  //struct context* ctx = (struct context *)context;
  /* fprintf(stderr, "\n=== NCCL Profiling Summary ===\n"); */
  /* fprintf(stderr, "AllReduce calls:       %" PRIu64 "\n", allReduceCount); */
  /* fprintf(stderr, "Broadcast calls:       %" PRIu64 "\n", broadcastCount); */
  /* fprintf(stderr, "Reduce calls:          %" PRIu64 "\n", reduceCount); */
  /* fprintf(stderr, "ReduceScatter calls:   %" PRIu64 "\n", reduceScatterCount); */
  /* fprintf(stderr, "AllGather calls:       %" PRIu64 "\n", allGatherCount); */
  /* fprintf(stderr, "AllToAll calls:        %" PRIu64 "\n", allToAllCount); */
  /* fprintf(stderr, "Unknown calls:         %" PRIu64 "\n", unknownCount); */
  /* fprintf(stderr, "=============================\n\n"); */

fprintf(stderr, "\n=== NCCL Profiling Summary ===\n");
fprintf(stderr, "%-18s %-12s %-20s\n", "Collective Type", "Calls", "Bytes Transferred");
fprintf(stderr, "---------------------------------------------------------\n");
fprintf(stderr, "%-18s %-12" PRIu64 " %-20" PRIu64 " Bytes\n", "AllReduce:", allReduceCount, allReduceBytes);
fprintf(stderr, "%-18s %-12" PRIu64 " %-20" PRIu64 " Bytes\n", "Broadcast:", broadcastCount, broadcastBytes);
fprintf(stderr, "%-18s %-12" PRIu64 " %-20" PRIu64 " Bytes\n", "Reduce:", reduceCount, reduceBytes);
fprintf(stderr, "%-18s %-12" PRIu64 " %-20" PRIu64 " Bytes\n", "ReduceScatter:", reduceScatterCount, reduceScatterBytes);
fprintf(stderr, "%-18s %-12" PRIu64 " %-20" PRIu64 " Bytes\n", "AllGather:", allGatherCount, allGatherBytes);
fprintf(stderr, "%-18s %-12" PRIu64 " %-20" PRIu64 " Bytes\n", "AllToAll:", allToAllCount, allToAllBytes);
fprintf(stderr, "%-18s %-12" PRIu64 " %-20" PRIu64 " Bytes\n", "Unknown:", unknownCount, unknownBytes);
fprintf(stderr, "=========================================================\n\n");


  /* fclose(ctx->debug_file); */
  /* free(ctx); */
  return ncclSuccess;
}



ncclResult_t Profiler_Event_Start(void* context, void** eHandle, ncclProfilerEventDescr_t* eDescr){

    *eHandle = NULL;
    struct context* ctx = (struct context*)context;

    if (eDescr->type == ncclProfileGroup) {
        /* struct group* event = (struct group*)malloc(sizeof(struct group)); */
        /* if (event == NULL) { */
        /*     return ncclInternalError; */
        /* } */
        // Get the next index in the group pool (circular buffer behavior)
        int index = __atomic_fetch_add(&ctx->groupIndex, 1, __ATOMIC_RELAXED) % GROUP_POOL_SIZE;
        struct group* event = &ctx->groupPool[index];
        event->ctx = ctx;
        event->type = ncclProfileGroup;
        event->startTs = gettime() - startTime;
        *eHandle = event;
    }

    if (eDescr->type == ncclProfileColl) {

        struct group* parent = (struct group *)eDescr->parentObj;
        if (parent == NULL) return ncclSuccess;

        int index = __atomic_fetch_add(&ctx->collIndex, 1, __ATOMIC_RELAXED) % COLL_POOL_SIZE;
        struct collective* event = &ctx->collPool[index];

        event->type = ncclProfileColl;
        size_t trafficBytes = eDescr->coll.trafficBytes;
        event->parent = (struct group*)eDescr->parentObj;
        event->name = eDescr->coll.func;
        event->startTs = gettime() - startTime;
        __atomic_fetch_add(&parent->refCount, 1, __ATOMIC_RELAXED);
        *eHandle = event;
        if (strcmp(eDescr->coll.func, "AllReduce") == 0) {
          __atomic_fetch_add(&allReduceCount, 1, __ATOMIC_RELAXED);
          __atomic_fetch_add(&allReduceBytes, trafficBytes, __ATOMIC_RELAXED);
        }
        else if (strcmp(eDescr->coll.func, "Broadcast") == 0) {
          __atomic_fetch_add(&broadcastCount, 1, __ATOMIC_RELAXED);
          __atomic_fetch_add(&broadcastBytes, trafficBytes, __ATOMIC_RELAXED);
        }
        else if (strcmp(eDescr->coll.func, "Reduce") == 0) {
          __atomic_fetch_add(&reduceCount, 1, __ATOMIC_RELAXED);
          __atomic_fetch_add(&reduceBytes, trafficBytes, __ATOMIC_RELAXED);
        }
        else if (strcmp(eDescr->coll.func, "ReduceScatter") == 0) {
          __atomic_fetch_add(&reduceScatterCount, 1, __ATOMIC_RELAXED);
          __atomic_fetch_add(&reduceScatterBytes, trafficBytes, __ATOMIC_RELAXED);
        }
        else if (strcmp(eDescr->coll.func, "AllGather") == 0) {
          __atomic_fetch_add(&allGatherCount, 1, __ATOMIC_RELAXED);
          __atomic_fetch_add(&allGatherBytes, trafficBytes, __ATOMIC_RELAXED);
        }
        else if (strcmp(eDescr->coll.func, "AllToAll") == 0) {
          __atomic_fetch_add(&allToAllCount, 1, __ATOMIC_RELAXED);
          __atomic_fetch_add(&allToAllBytes, trafficBytes, __ATOMIC_RELAXED);
        }
        // Keeping track of this for now for debugging purposes
        else {
          __atomic_fetch_add(&unknownCount, 1, __ATOMIC_RELAXED);
          __atomic_fetch_add(&unknownBytes, trafficBytes, __ATOMIC_RELAXED);
        }
    } else if (eDescr->type == ncclProfileProxyOp) {
      // We will need to change this for p2p
      struct collective* eventBase = (struct collective *)eDescr->parentObj;
      if (eventBase == NULL) return ncclSuccess;
      if ( eDescr->proxyOp.pid != pid ){
        struct proxyOp* event;
        int detachId = __atomic_fetch_add(&detachPoolIndex, 1, __ATOMIC_RELAXED) % detachPoolSize;
        event = &detachPool[detachId];
        event->type = ncclProfileProxyOp;
        event->pid = eDescr->proxyOp.pid;
        event->parent = NULL;
        *eHandle = event;
        return ncclSuccess;
      }
    }

    if (eventBase->type == ncclProfileColl) {
      // Cannot be NULL
      struct collective* parent = (struct collective *)eDescr->parentObj;
      event->type = ncclProfileProxyOp;
      event->pid = eDescr->proxyOp.pid;
      event->parent = parent;
      __atomic_fetch_add(&parent->refCount, 1, __ATOMIC_RELAXED);
      *eHandle = event;
    }


    return ncclSuccess;
}



static void updateEvent(void* handle) {
    if (handle == NULL) return;

    uint8_t type = *(uint8_t*)handle;
    if (type == ncclProfileGroup){
      struct group* event = (struct group *)handle;
      if (__atomic_sub_fetch(&event->refCount, 1, __ATOMIC_RELAXED) == 0) {
        event->stopTs = gettime() - startTime;
        // return group event to the pool
        //__atomic_fetch_add(&event->ctx->groupPoolBase, 1, __ATOMIC_RELAXED);
      }
    } else if {
      struct collective* event = (struct collective *)handle;
      if (__atomic_sub_fetch(&event->base.refCount, 1, __ATOMIC_RELAXED) == 0) {
        event->base.stopTs = gettime() - startTime;
        double duration = event->base.stopTs - event->base.startTs;
        if (strcmp(eDescr->coll.func, "AllReduce") == 0) {
          __atomic_fetch_add(&allReduceTime, duration, __ATOMIC_RELAXED);
        }
        else if (strcmp(eDescr->coll.func, "Broadcast") == 0) {
          __atomic_fetch_add(&broadcastTime, duration, __ATOMIC_RELAXED);
        }
        else if (strcmp(eDescr->coll.func, "Reduce") == 0) {
          __atomic_fetch_add(&reduceTime, duration, __ATOMIC_RELAXED);
        }
        else if (strcmp(eDescr->coll.func, "ReduceScatter") == 0) {
          __atomic_fetch_add(&reduceScatterTime, duration, __ATOMIC_RELAXED);
        }
        else if (strcmp(eDescr->coll.func, "AllGather") == 0) {
          __atomic_fetch_add(&allGatherTime, duration, __ATOMIC_RELAXED);
        }
        else if (strcmp(eDescr->coll.func, "AllToAll") == 0) {
          __atomic_fetch_add(&allToAllTime, duration, __ATOMIC_RELAXED);
        }
        // Keeping track of this for now for debugging purposes
        else {
          __atomic_fetch_add(&unknownTime, duration, __ATOMIC_RELAXED);
        }
        updateEvent(event->base.parent);
        return;
      }
    }
    } else if (type == ncclProfileProxyOp) {
      struct proxyOp* event = (struct proxyOp *)handle;
      if (__atomic_sub_fetch(&event->parent->refCount, 1, __ATOMIC_RELAXED) == 0) {
        event->stopTs = gettime() - startTime;
        // return proxy event to the pool
        //__atomic_fetch_add(&event->ctx->proxyPoolBase, 1, __ATOMIC_RELAXED);
      }
    }
}

__hidden ncclResult_t Profiler_Event_Stop(void* eHandle) {
  if (eHandle == NULL) return ncclSuccess;
  uint8_t type = *(uint8_t*)eHandle;

  if (type == ncclProfileGroup) {
    struct group* g = (struct group*) eHandle;
    g->stopTs = getTime() - startTime;
    return ncclSuccess;
  }
  else if (type == ncclProfileColl) {
    struct collective* c = (struct collective*) eHandle;
    c->stopTs = getTime() -startTime;
    return ncclSuccess;
  }
  else if (type == ncclProfileProxyOp) {
    struct proxyOp* p = (struct proxyOp*) eHandle;
    p->stopTs = getTime()- startTime;
    // Now that the proxy op is done, finalize it by calling updateEvent
    updateEvent(eHandle);
    return ncclSuccess;
  }

  // If other types exist:
  // updateEvent(eHandle);
  return ncclSuccess;
}


__hidden ncclResult_t Profiler_Event_Record(void* eHandle, ncclProfilerEventState_t eState, ncclProfilerEventStateArgs_t* eStateArgs){
  return ncclSuccess;
}



ncclProfiler_t ncclProfiler_v2 = {
   .name = plugin_name,
   .init = Profiler_Init,
   .startEvent = Profiler_Event_Start,
   .stopEvent = Profiler_Event_Stop,
   .recordEventState = Profiler_Event_Record,
   .finalize = Profiler_Finalize
};
