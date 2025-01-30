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


static struct {
    uint64_t count;
    uint64_t bytes;
    double time;
} stats[nccl_num_colls] = {0};

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
  enum nccl_colls name;   // Index in the collective name array
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
static FILE *debug_file = NULL;
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
  // Dummy loop to let some time pass
  for (int i = 0; i < 10000; i++) total += __rdtsc();

  gettimeofday(&tv, NULL);

  timeCycles = __rdtsc() - timeCycles;  // Compute elapsed cycles
  time += tv.tv_sec*1e6 + tv.tv_usec;  // Compute elapsed real-world time
  freq = timeCycles / time;
}

// returns current timestamp in useconds
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


  fprintf(stderr, "Profiler_Init: %s\n",plugin_name);
  fprintf(stderr, "Profiler_Init: eActivationMask = %d\n", *eActivationMask);

  char debug_file_name[64];
  snprintf(debug_file_name, 64, "./fools_debug_%d.log", pid);
  debug_file = fopen(debug_file_name, "a+");
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

/* fprintf(stderr, "\n=== NCCL Profiling Summary ===\n"); */
/* fprintf(stderr, "%-18s %-12s %-20s\n", "Collective Type", "Calls", "Bytes Transferred"); */
/* fprintf(stderr, "---------------------------------------------------------\n"); */
/* fprintf(stderr, "%-18s %-12" PRIu64 " %-20" PRIu64 " Bytes\n", "AllReduce:", allReduceCount, allReduceBytes); */
/* fprintf(stderr, "%-18s %-12" PRIu64 " %-20" PRIu64 " Bytes\n", "Broadcast:", broadcastCount, broadcastBytes); */
/* fprintf(stderr, "%-18s %-12" PRIu64 " %-20" PRIu64 " Bytes\n", "Reduce:", reduceCount, reduceBytes); */
/* fprintf(stderr, "%-18s %-12" PRIu64 " %-20" PRIu64 " Bytes\n", "ReduceScatter:", reduceScatterCount, reduceScatterBytes); */
/* fprintf(stderr, "%-18s %-12" PRIu64 " %-20" PRIu64 " Bytes\n", "AllGather:", allGatherCount, allGatherBytes); */
/* fprintf(stderr, "%-18s %-12" PRIu64 " %-20" PRIu64 " Bytes\n", "AllToAll:", allToAllCount, allToAllBytes); */
/* fprintf(stderr, "%-18s %-12" PRIu64 " %-20" PRIu64 " Bytes\n", "Unknown:", unknownCount, unknownBytes); */
/* fprintf(stderr, "=========================================================\n\n"); */

  fprintf(stderr, "\n=================== NCCL PROFILING SUMMARY ===================\n");
  fprintf(stderr, "%-18s %-12s %-15s %-15s\n",
          "Collective Type", "Calls", "Bytes Transferred", "Total Time (us)");
  fprintf(stderr, "--------------------------------------------------------------------------\n");

  for (int i = 0; i < nccl_num_colls; i++) {
    fprintf(stderr, "%-18s %-12" PRIu64 " %-15" PRIu64 " %-15.6f\n",
            nccl_coll_names[i], stats[i].count, stats[i].bytes, stats[i].time);
  }

  fprintf(stderr, "==========================================================================\n\n");


  struct context* ctx = (struct context*)context;
  free(ctx);
  if (debug_file) fclose(debug_file);
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
        const char* name = eDescr->coll.func;
        event->startTs = gettime() - startTime;
        __atomic_fetch_add(&parent->refCount, 1, __ATOMIC_RELAXED);
        *eHandle = event;
        if (strcmp(name, "AllReduce") == 0) {
          event->name = nccl_allreduce;
          /* __atomic_fetch_add(&allReduceCount, 1, __ATOMIC_RELAXED); */
          /* __atomic_fetch_add(&allReduceBytes, trafficBytes, __ATOMIC_RELAXED); */
        }
        else if (strcmp(name, "Broadcast") == 0) {
          event->name = nccl_broadcast;
          /* __atomic_fetch_add(&broadcastCount, 1, __ATOMIC_RELAXED); */
          /* __atomic_fetch_add(&broadcastBytes, trafficBytes, __ATOMIC_RELAXED); */
        }
        else if (strcmp(name, "Reduce") == 0) {
          event->name = nccl_reduce;
          /* __atomic_fetch_add(&reduceCount, 1, __ATOMIC_RELAXED); */
          /* __atomic_fetch_add(&reduceBytes, trafficBytes, __ATOMIC_RELAXED); */
        }
        else if (strcmp(name, "ReduceScatter") == 0) {
          event->name = nccl_reduce_scatter;
          /* __atomic_fetch_add(&reduceScatterCount, 1, __ATOMIC_RELAXED); */
          /* __atomic_fetch_add(&reduceScatterBytes, trafficBytes, __ATOMIC_RELAXED); */
        }
        else if (strcmp(name, "AllGather") == 0) {
          event->name = nccl_allgather;
          /* __atomic_fetch_add(&allGatherCount, 1, __ATOMIC_RELAXED); */
          /* __atomic_fetch_add(&allGatherBytes, trafficBytes, __ATOMIC_RELAXED); */
        }
        else if (strcmp(name, "AllToAll") == 0) {
          event->name = nccl_alltoall;
          /* __atomic_fetch_add(&allToAllCount, 1, __ATOMIC_RELAXED); */
          /* __atomic_fetch_add(&allToAllBytes, trafficBytes, __ATOMIC_RELAXED); */
        }
        // Keeping track of this for now for debugging purposes
        else {
          event->name = nccl_unknown;
          /* __atomic_fetch_add(&unknownCount, 1, __ATOMIC_RELAXED); */
          /* __atomic_fetch_add(&unknownBytes, trafficBytes, __ATOMIC_RELAXED); */
        }
        // It is better to update those now so we dont carry them around
        __atomic_fetch_add(&stats[event->name].count, 1, __ATOMIC_RELAXED);
        __atomic_fetch_add(&stats[event->name].bytes, trafficBytes, __ATOMIC_RELAXED);
    } else if (eDescr->type == ncclProfileProxyOp) {
      // We will need to change this for p2p
      struct collective* eventBase = (struct collective *)eDescr->parentObj;
      if (eventBase == NULL) return ncclSuccess;
      fprintf(debug_file, "ProxyOp\n");
      if ( eDescr->proxyOp.pid != pid ){
        int index = __atomic_fetch_add(&ctx->proxyIndex, 1, __ATOMIC_RELAXED) % PROXY_POOL_SIZE;
        struct proxyOp* event = &ctx->proxyPool[index];
        event->type = ncclProfileProxyOp;
        event->pid = eDescr->proxyOp.pid;
        event->parent = NULL;
        *eHandle = event;
        return ncclSuccess;
      }

    if (eventBase->type == ncclProfileColl) {
      // Cannot be NULL
      struct collective* parent = (struct collective *)eDescr->parentObj;
      int index = __atomic_fetch_add(&ctx->proxyIndex, 1, __ATOMIC_RELAXED) % PROXY_POOL_SIZE;
      struct proxyOp* event = &ctx->proxyPool[index];
      event->type = ncclProfileProxyOp;
      event->pid = eDescr->proxyOp.pid;
      event->parent = parent;
      __atomic_fetch_add(&parent->refCount, 1, __ATOMIC_RELAXED);
      *eHandle = event;
    }
    }

    return ncclSuccess;
}



static void updateEvent(void* handle) {
    if (handle == NULL) return;

    uint8_t type = *(uint8_t*)handle;
    if (type == ncclProfileGroup){
      struct group* event = (struct group *)handle;
      __atomic_sub_fetch(&event->refCount, 1, __ATOMIC_RELAXED);
      // We are not measuring group time yet
      /* if (__atomic_sub_fetch(&event->refCount, 1, __ATOMIC_RELAXED) == 0) { */
      /*    event->stopTs = gettime() - startTime; */
      /* } */
    } else if (type == ncclProfileColl) {
      struct collective* event = (struct collective *)handle;
      if (__atomic_sub_fetch(&event->refCount, 1, __ATOMIC_RELAXED) == 0) {
        event->stopTs = gettime() - startTime;
        double duration = event->stopTs - event->startTs;
        // Update the time in stats
        fprintf(debug_file, "Collective %s took %lf us\n", nccl_coll_names[event->name], duration);
        stats[event->name].time += duration;
        updateEvent(event->parent);
        return;
      }
    } else if (type == ncclProfileProxyOp) {
      struct proxyOp* event = (struct proxyOp *)handle;
      // We are not measuring proxy ops time yet
      updateEvent(event->parent);
    }
}

__hidden ncclResult_t Profiler_Event_Stop(void* eHandle) {
  if (eHandle == NULL) return ncclSuccess;
  uint8_t type = *(uint8_t*)eHandle;

  if (type == ncclProfileGroup) {
    //struct group* event = (struct group*) eHandle;
    //event->stopTs = getTime() - startTime;
    return ncclSuccess;
  }
  else if (type == ncclProfileColl) {
    struct collective* event = (struct collective*) eHandle;
    event->stopTs = gettime() -startTime;
    double duration = event->stopTs - event->startTs;
    stats[event->name].time += duration;
    return ncclSuccess;
  }
  else if (type == ncclProfileProxyOp) {
    //struct proxyOp* event = (struct proxyOp*) eHandle;
    // event->stopTs = getTime()- startTime;
    // Now that the proxy op is done, finalize it by calling updateEvent
    updateEvent(eHandle);
    return ncclSuccess;
  }

  // If other types exist such as ncclProfileP2p, we can add them here
  updateEvent(eHandle);
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
