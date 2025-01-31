#include <stdio.h>
#include <pthread.h>
#include <string.h>
#include <linux/limits.h>
#include <sys/time.h>
#include <sys/types.h>
#include <sys/syscall.h>
#include <unistd.h>
#include <inttypes.h>
#include <stdbool.h>
#include <x86intrin.h>
#include <stdatomic.h>
#include "profiler.h"


#define __hidden __attribute__ ((visibility("hidden")))
#define GROUP_POOL_SIZE 64
#define COLL_POOL_SIZE 64
#define P2P_POOL_SIZE 64
#define PROXY_POOL_SIZE 64

#define MAX_CHANNELS                     32
#define MAX_STEPS                        16
#define MAX_OPS                          16 // Up to 64K ranks for PAT


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
    "Unknown_Collective",
};


enum nccl_p2p {
    nccl_p2p_send,
    nccl_p2p_recv,
    nccl_p2p_unknown, // For unexpected cases
    nccl_num_p2p // Keeps track of total primitives
};

static const char* nccl_p2p_names[nccl_num_p2p] = {
    "Send",
    "Recv",
    "Uknown_P2P"
};

/* static const int groupPoolSize = 128; */
/* static const int collPoolSize = 128; */


static struct {
    uint64_t count;
    uint64_t bytes;
    _Atomic double time;
  // We may add more things here
} stats[nccl_num_colls] = {0};


static struct {
    uint64_t count;
    uint64_t typecount;
    _Atomic double time;
} stats_p2p[nccl_num_p2p] = {0};

static struct {
    uint64_t count;
    _Atomic double time;
} stats_group = {0};

struct context;

struct group {
  uint8_t type;
  struct context* ctx;
  int refCount;
  double startTs;
  double stopTs;
};


// task level event base structure
struct taskEventBase {
  uint8_t type;                     // event type: collective/p2p
  //int rank;                         // rank of the operation in NCCL communicator
  //uint64_t commHash;                // communicator identifier
  const char* func;                 // ncclFunc*
  int refCount;                     // number of references for this operation
  struct group* parent;             // parent event group
  double startTs;
  double stopTs;
};


struct proxyOp {
  uint8_t type;           // ncclProfileProxyOp
  pid_t pid;
  struct taskEventBase* parent;     // parent event p2p/collective
};

struct collective {
  struct taskEventBase base;
  enum nccl_colls name;   // Index in the collective name array
  // struct proxyOp send[MAX_CHANNELS][MAX_OPS];// array of send proxy operation events
  // struct proxyOp recv[MAX_CHANNELS][MAX_OPS];// array of recv proxy operation events
  // int nProxyOps[MAX_CHANNELS];
};

struct p2p {
  struct taskEventBase base;
  enum nccl_p2p name;
  // struct proxyOp op[MAX_CHANNELS];
};



struct context {
 int groupIndex;
 struct group groupPool[GROUP_POOL_SIZE];
 int collIndex;
 struct collective collPool[COLL_POOL_SIZE];
 int p2pIndex;
 struct p2p p2pPool[P2P_POOL_SIZE];
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



void atomic_add_double(_Atomic double *target, double increment) {
    double current;
    double desired;
    do {
        current = atomic_load(target);
        desired = current + increment;
    } while (!atomic_compare_exchange_weak(target, &current, desired));
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


  /* fprintf(stderr, "Profiler_Init: %s\n",plugin_name); */
  /* fprintf(stderr, "Profiler_Init: eActivationMask = %d\n", *eActivationMask); */
#ifdef DEBUG
  char debug_file_name[64];
  snprintf(debug_file_name, 64, "./fools_debug_%d.log", pid);
  debug_file = fopen(debug_file_name, "a+");
#endif
  // Allocate memory for the context
  struct context* ctx = (struct context*)calloc(1, sizeof(struct context));
  if (ctx == NULL) {
      fprintf(stderr, "Profiler_Init: Failed to allocate memory for context\n");
      return ncclInternalError; // Return an appropriate NCCL error code
  }
  ctx->groupIndex = 0;
  ctx->collIndex = 0;
  ctx->p2pIndex = 0;
  ctx->proxyIndex = 0;
  // Assign the context to the output parameter
  *context = ctx;


  return ncclSuccess;
}

__hidden ncclResult_t Profiler_Finalize(void* context) {

  fprintf(stderr, "\n=================== NCCL PROFILING SUMMARY ===================\n");

  fprintf(stderr, "%-18s %-12s %-20s %-15s\n",
          "Collective Type", "Calls", "Bytes Transferred", "Total Time (us)");
  fprintf(stderr, "--------------------------------------------------------------------------\n");

  for (int i = 0; i < nccl_num_colls; i++) {
    if (stats[i].count == 0) continue;
    fprintf(stderr, "%-18s %-12" PRIu64 " %-20" PRIu64 " %-15.6f\n",
            nccl_coll_names[i], stats[i].count, stats[i].bytes, stats[i].time);
  }
  fprintf(stderr, "%-18s %-12" PRIu64 " %-20" PRIu64 " %-15.6f\n","Group", stats_group.count, (uint64_t)0, stats_group.time);


  fprintf(stderr, "==========================================================================\n\n");


  struct context* ctx = (struct context*)context;
  free(ctx);
  if (debug_file) fclose(debug_file);
  return ncclSuccess;
}



ncclResult_t Profiler_Event_Start(void* context, void** eHandle, ncclProfilerEventDescr_t* eDescr){
  *eHandle = NULL;
  struct context* ctx = (struct context*)context;
#ifdef DEBUG
  fprintf(debug_file, "Profiler_Event_Start: %d\n", eDescr->type);
  fflush(debug_file);
#endif

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
        __atomic_fetch_add(&stats_group.count, 1, __ATOMIC_RELAXED);
        event->startTs = gettime();
        *eHandle = event;
    } else if (eDescr->type == ncclProfileColl) {

        struct group* parent = (struct group *)eDescr->parentObj;
        if (parent == NULL) return ncclSuccess;

        int index = __atomic_fetch_add(&ctx->collIndex, 1, __ATOMIC_RELAXED) % COLL_POOL_SIZE;
        struct collective* event = &ctx->collPool[index];

        event->base.type = ncclProfileColl;
        size_t trafficBytes = eDescr->coll.trafficBytes;
        event->base.parent = parent;
        const char* name = eDescr->coll.func;
        __atomic_fetch_add(&parent->refCount, 1, __ATOMIC_RELAXED);
        if (strcmp(name, "AllReduce") == 0) {
          event->name = nccl_allreduce;
        }
        else if (strcmp(name, "Broadcast") == 0) {
          event->name = nccl_broadcast;
        }
        else if (strcmp(name, "Reduce") == 0) {
          event->name = nccl_reduce;
        }
        else if (strcmp(name, "ReduceScatter") == 0) {
          event->name = nccl_reduce_scatter;
        }
        else if (strcmp(name, "AllGather") == 0) {
          event->name = nccl_allgather;
        }
        else if (strcmp(name, "AllToAll") == 0) {
          event->name = nccl_alltoall;
        }
        // Keeping track of this for now for debugging purposes
        else {
          event->name = nccl_unknown;
        }
        // It is better to update those now so we dont carry them around
        __atomic_fetch_add(&stats[event->name].count, 1, __ATOMIC_RELAXED);
        __atomic_fetch_add(&stats[event->name].bytes, trafficBytes, __ATOMIC_RELAXED);
        event->base.startTs = gettime();
        *eHandle = event;
    } else if (eDescr->type == ncclProfileP2p) {
      struct group* parent = (struct group *)eDescr->parentObj;
      if (parent == NULL) return ncclSuccess;
      int index = __atomic_fetch_add(&ctx->p2pIndex, 1, __ATOMIC_RELAXED) % P2P_POOL_SIZE;
      struct p2p* event = &ctx->p2pPool[index];
      event->base.type = ncclProfileP2p;
      event->base.parent = parent;
      if (strcmp(eDescr->p2p.func, "Send") == 0) {
        event->name = nccl_p2p_send;
      }
      else if (strcmp(eDescr->p2p.func, "Recv") == 0) {
        event->name = nccl_p2p_recv;
      }
      else {
        event->name = nccl_p2p_unknown;
      }
      __atomic_fetch_add(&stats_p2p[event->name].count, 1, __ATOMIC_RELAXED);
      __atomic_fetch_add(&stats_p2p[event->name].typecount, eDescr->p2p.count, __ATOMIC_RELAXED);
      event->base.startTs = gettime();
      *eHandle = event;

    } else if (eDescr->type ==  ncclProfileProxyOp) {

      //fprintf(debug_file, "ProxyOp\n");
      struct taskEventBase* eventBase = (struct taskEventBase *)eDescr->parentObj;
      if (eventBase == NULL) return ncclSuccess;
      //fprintf(debug_file, "ProxyOp parent not NULL\n");
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
        event->parent = eventBase;
        __atomic_fetch_add(&parent->base.refCount, 1, __ATOMIC_RELAXED);
        *eHandle = event;
      }
      else{
        struct p2p* parent = (struct p2p *)eDescr->parentObj;
        int index = __atomic_fetch_add(&ctx->proxyIndex, 1, __ATOMIC_RELAXED) % PROXY_POOL_SIZE;
        struct proxyOp* event = &ctx->proxyPool[index];
        event->type = ncclProfileProxyOp;
        event->pid = eDescr->proxyOp.pid;
        event->parent = eventBase;
        __atomic_fetch_add(&parent->base.refCount, 1, __ATOMIC_RELAXED);
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
      if (__atomic_sub_fetch(&event->refCount, 1, __ATOMIC_RELAXED) == 0) {
         event->stopTs = gettime();
         double duration = event->stopTs - event->startTs;
         stats_group.time += duration;
      }
    } else if (type == ncclProfileColl) {
      struct collective* event = (struct collective *)handle;
      if (__atomic_sub_fetch(&event->base.refCount, 1, __ATOMIC_RELAXED) == 0) {
        event->base.stopTs = gettime();
        double duration = event->base.stopTs - event->base.startTs;
        // Update the time in stats
        //fprintf(debug_file, "Collective %s took %lf us\n", nccl_coll_names[event->name], duration);
        stats[event->name].time += duration;
        updateEvent(event->base.parent);
        return;
      }
    } else if ( type == ncclProfileP2p ) {
      struct p2p* event = (struct p2p *)handle;
      if (__atomic_sub_fetch(&event->base.refCount, 1, __ATOMIC_RELAXED) == 0) {
        event->base.stopTs = gettime();
        double duration = event->base.stopTs - event->base.startTs;
        // Update the time in stats
        //fprintf(debug_file, "P2P %s took %lf us\n", nccl_p2p_names[event->name], duration);
        stats_p2p[event->name].time += duration;
        updateEvent(event->base.parent);
        return;
      }
    }
    else if (type == ncclProfileProxyOp) {
      struct proxyOp* event = (struct proxyOp *)handle;
      // We are not measuring proxy ops time yet
      updateEvent(event->parent);
    }
}

__hidden ncclResult_t Profiler_Event_Stop(void* eHandle) {
  if (eHandle == NULL) return ncclSuccess;

  uint8_t type = *(uint8_t*)eHandle;
#ifdef DEBUG
  fprintf(debug_file, "Profiler_Event_Stop: %d\n", type);
  fflush(debug_file);
#endif

  if (type == ncclProfileGroup) {
    struct group* event = (struct group*) eHandle;
    event->stopTs = gettime();
    // Update the time in stats atomically
    atomic_add_double(&stats_group.time, event->stopTs - event->startTs);
#ifdef DEBUG
    fprintf(debug_file, "Group took %lf us, Accumulated %lf\n", event->stopTs - event->startTs, stats_group.time);
    fflush(debug_file);
#endif
    // Update the time in case proxy ops are used
    //event->startTs = event->stopTs;
    return ncclSuccess;
  }
  else if (type == ncclProfileColl) {
    struct collective* event = (struct collective*) eHandle;
    event->base.stopTs = gettime();
    // Update the time in collective stats atomically
    atomic_add_double(&stats[event->name].time, event->base.stopTs - event->base.startTs);
    // Update the time in case proxy ops are used
    // event->base.startTs = event->base.stopTs;
    return ncclSuccess;
  }
  else if (type == ncclProfileP2p) {
    struct p2p* event = (struct p2p*) eHandle;
    event->base.stopTs = gettime();
    stats_p2p[event->name].time +=  event->base.stopTs - event->base.startTs;
    // Update the time in case proxy ops are used
    //event->base.startTs = event->base.stopTs;
    return ncclSuccess;
  }

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
