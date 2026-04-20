// NVSHMEM in a non-primary CUDA context -- minimal reproducer.
// See README.md for the full explanation.

#include <cstdio>
#include <cstdlib>
#include <cuda.h>
#include <cuda_runtime.h>
#include <nvshmem.h>
#include <nvshmemx.h>

#define CK(e)  do { cudaError_t _=(e); if (_) { std::fprintf(stderr,"CUDA %s:%d %s\n", \
    __FILE__,__LINE__,cudaGetErrorString(_)); std::exit(2);} } while(0)
#define CKU(e) do { CUresult _=(e); if (_) { const char*n=nullptr,*m=nullptr; \
    cuGetErrorName(_,&n); cuGetErrorString(_,&m); \
    std::fprintf(stderr,"CU %s:%d %s (%s)\n",__FILE__,__LINE__,n?n:"?",m?m:"?"); std::exit(2);} } while(0)

__global__ void write_local(int* s, int mype) { s[mype] = mype; }
__global__ void put_peer (int* s, int mype, int peer) { nvshmem_int_p(&s[mype], 100+mype, peer); }

int main() {
    int pre_rank = -1;
    for (const char* n : {"PMI_RANK","PMIX_RANK","OMPI_COMM_WORLD_RANK","SLURM_PROCID"}) {
        if (const char* v = std::getenv(n)) { pre_rank = std::atoi(v); break; }
    }

    CKU(cuInit(0));
    int ngpus = 0; CKU(cuDeviceGetCount(&ngpus));
    CUdevice dev; CKU(cuDeviceGet(&dev, pre_rank % ngpus));

    // Fresh non-primary context, made current by cuCtxCreate.
    CUcontext my_ctx = nullptr, prim_ctx = nullptr;
    CUctxCreateParams params{};
    CKU(cuCtxCreate(&my_ctx, &params, CU_CTX_SCHED_AUTO, dev));
    CKU(cuDevicePrimaryCtxRetain(&prim_ctx, dev));
    std::printf("PE %d: my_ctx=%p primary=%p (distinct? %s)\n",
                pre_rank, (void*)my_ctx, (void*)prim_ctx,
                my_ctx != prim_ctx ? "yes" : "NO!");

    std::printf("PE %d: before nvshmem_init\n", pre_rank);
    nvshmem_init();
    std::printf("PE %d: after  nvshmem_init  <-- not printed if NVSHMEM aborted inside init\n", pre_rank);

    const int mype = nvshmem_my_pe();
    const int npes = nvshmem_n_pes();
    if (npes != 2) { nvshmem_finalize(); return 1; }
    const int peer = 1 - mype;

    int* sym = static_cast<int*>(nvshmem_malloc(npes * sizeof(int)));
    if (!sym) { nvshmem_finalize(); return 3; }

    CK(cudaMemset(sym, 0, npes * sizeof(int)));
    nvshmem_barrier_all();
    write_local<<<1,1>>>(sym, mype); CK(cudaDeviceSynchronize());
    nvshmem_barrier_all();
    put_peer<<<1,1>>>(sym, mype, peer); CK(cudaDeviceSynchronize());
    nvshmem_barrier_all();

    int h[8]{};
    CK(cudaMemcpy(h, sym, npes*sizeof(int), cudaMemcpyDeviceToHost));
    std::printf("PE %d: slots=[%d,%d] expect=[%d,%d]\n",
                mype, h[0], h[1],
                mype==0 ? 0   : 100,
                mype==0 ? 101 : 1);

    nvshmem_free(sym);
    nvshmem_finalize();
    CKU(cuDevicePrimaryCtxRelease(dev));
    CKU(cuCtxDestroy(my_ctx));
    return 0;
}
