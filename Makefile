#
# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
#
# See LICENSE.txt for license information
#
NCCL_HOME := /home/staff/vardas/nccl/build
INC := -I$(NCCL_HOME)/include -I/home/staff/vardas/nccl/ext-profiler/example/nccl -I$(CUDA_HOME)/include -Inccl
PLUGIN_SO := libnccl-profiler.so

default: $(PLUGIN_SO)

$(PLUGIN_SO): example.c
	$(CC) $(INC) -Wall -Wextra -std=c11 -g -fPIC -shared -o $@ -Wl,-soname,$(PLUGIN_SO) $^

clean:
	rm -f $(PLUGIN_SO)
