NCCL_HOME := /home/staff/vardas/nccl/build
LIBS := -L$(CUDA_HOME)/lib64 -L/home/staff/vardas/spack/opt/spack/linux-debian11-zen2/gcc-10.2.1/cuda-11.8.0-6znhkjhz2vquzikacb3hbato2llnj3qi/extras/CUPTI/lib64 -L$(NCCL_HOME)/lib
INC := -I/home/staff/vardas/spack/opt/spack/linux-debian11-zen2/gcc-10.2.1/cuda-11.8.0-6znhkjhz2vquzikacb3hbato2llnj3qi/extras/CUPTI/include/ -I$(NCCL_HOME)/include -I/home/staff/vardas/nccl/ext-profiler/example/nccl -I$(CUDA_HOME)/include
PLUGIN_SO := libnccl-profiler.so
CFLAGS = -Wall -Wextra -std=c11 -O3 #-DDEBUG

default: $(PLUGIN_SO)

$(PLUGIN_SO): example.c
	$(CC) $(INC) $(LIBS) $(CFLAGS) -g -fPIC -shared -o $@ -Wl,-soname,$(PLUGIN_SO) $^ -lnccl -lcudart -lcupti -latomic -pthread

clean:
	rm -f $(PLUGIN_SO)
