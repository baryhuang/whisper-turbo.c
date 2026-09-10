#define _GNU_SOURCE
/* Read-only memory observation for InstaCloud's namespace root. memory.peak is
 * lifetime-to-date, NOT a reset per-run counter. A prior higher peak makes a
 * run inconclusive; it can never manufacture a lower peak. */
#include <errno.h>
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/resource.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/wait.h>
#include <time.h>
#include <unistd.h>
static unsigned long long number(const char *path) {
    unsigned long long value;FILE *f=fopen(path,"r");
    if(!f||fscanf(f,"%llu",&value)!=1){perror(path);exit(2);}
    fclose(f);return value;
}
static double now(void){struct timespec t;clock_gettime(CLOCK_MONOTONIC,&t);return t.tv_sec+t.tv_nsec*1e-9;}
static int evict(const char *path,size_t *warm_pages) {
    int model=open(path,O_RDONLY|O_CLOEXEC);struct stat st;
    if(model<0||fstat(model,&st)||st.st_size<=0){perror(path);return 2;}
    int advised=posix_fadvise(model,0,0,POSIX_FADV_DONTNEED);
    if(advised){errno=advised;perror("evict model cache");return 2;}
    size_t pages=((size_t)st.st_size+(size_t)getpagesize()-1)/(size_t)getpagesize();
    unsigned char *resident=malloc(pages);
    void *mapping=mmap(NULL,(size_t)st.st_size,PROT_READ,MAP_PRIVATE,model,0);
    if(!resident||mapping==MAP_FAILED||mincore(mapping,(size_t)st.st_size,resident))return 2;
    size_t warm=0;for(size_t i=0;i<pages;i++)warm+=(resident[i]&1)!=0;
    munmap(mapping,(size_t)st.st_size);close(model);free(resident);
    if(warm>1){fprintf(stderr,"model cache eviction incomplete for %s: %zu resident pages\n",path,warm);return 2;}
    *warm_pages+=warm;
    return 0;
}
int main(int argc,char **argv) {
    int first=1,models=1,command=2;
    if(argc>2&&!strcmp(argv[1],"--models")){
        char *end=NULL;long count=strtol(argv[2],&end,10);
        if(!end||*end||count<1||count>32)return 2;
        first=3;models=(int)count;command=first+models;
    }
    if(argc<=command){fprintf(stderr,"usage: observe MODEL COMMAND [ARG...]\n       observe --models N MODEL... COMMAND [ARG...]\n");return 2;}
    size_t warm=0;
    for(int i=0;i<models;++i)if(evict(argv[first+i],&warm))return 2;
    FILE *swaps=fopen("/proc/swaps","r");char line[512];
    if(!swaps||!fgets(line,sizeof(line),swaps))return 2;
    if(fgets(line,sizeof(line),swaps)){fprintf(stderr,"swap is configured; refusing benchmark\n");return 2;}
    fclose(swaps);
    unsigned long long before=number("/sys/fs/cgroup/memory.peak");
    unsigned long long initial=number("/sys/fs/cgroup/memory.current");
    double start=now();pid_t child=fork();if(child<0)return 2;
    if(!child){execvp(argv[command],argv+command);perror("exec");_exit(127);}
    int status;struct rusage usage;
    while(wait4(child,&status,0,&usage)<0)if(errno!=EINTR)return 2;
    double seconds=now()-start;
    unsigned long long after=number("/sys/fs/cgroup/memory.peak");
    int code=WIFEXITED(status)?WEXITSTATUS(status):128+WTERMSIG(status);
    fprintf(stderr,"OBSERVATION {\"counter\":\"execution_cgroup_lifetime_peak\",\"prior_peak_bytes\":%llu,\"initial_current_bytes\":%llu,\"peak_bytes\":%llu,\"peak_rss_bytes\":%llu,\"seconds\":%.6f,\"exit_code\":%d,\"model_count\":%d,\"model_initial_resident_pages\":%zu,\"swap_configured\":false}\n",
        before,initial,after,(unsigned long long)usage.ru_maxrss*1024,seconds,code,models,warm);
    return code;
}
