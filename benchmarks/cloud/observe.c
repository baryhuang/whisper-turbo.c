#define _GNU_SOURCE
/* Read-only memory observation for InstaCloud's namespace root. memory.peak is
 * lifetime-to-date, NOT a reset per-run counter. A prior higher peak makes a
 * run inconclusive; it can never manufacture a lower peak. */
#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/resource.h>
#include <sys/wait.h>
#include <time.h>
#include <unistd.h>
static unsigned long long number(const char *path) {
    unsigned long long value;FILE *f=fopen(path,"r");
    if(!f||fscanf(f,"%llu",&value)!=1){perror(path);exit(2);}
    fclose(f);return value;
}
static double now(void){struct timespec t;clock_gettime(CLOCK_MONOTONIC,&t);return t.tv_sec+t.tv_nsec*1e-9;}
int main(int argc,char **argv) {
    if(argc<2)return 2;
    FILE *swaps=fopen("/proc/swaps","r");char line[512];
    if(!swaps||!fgets(line,sizeof(line),swaps))return 2;
    if(fgets(line,sizeof(line),swaps)){fprintf(stderr,"swap is configured; refusing benchmark\n");return 2;}
    fclose(swaps);
    unsigned long long before=number("/sys/fs/cgroup/memory.peak");
    unsigned long long initial=number("/sys/fs/cgroup/memory.current");
    double start=now();pid_t child=fork();if(child<0)return 2;
    if(!child){execvp(argv[1],argv+1);perror("exec");_exit(127);}
    int status;struct rusage usage;
    while(wait4(child,&status,0,&usage)<0)if(errno!=EINTR)return 2;
    double seconds=now()-start;
    unsigned long long after=number("/sys/fs/cgroup/memory.peak");
    int code=WIFEXITED(status)?WEXITSTATUS(status):128+WTERMSIG(status);
    fprintf(stderr,"OBSERVATION {\"counter\":\"whole_service_lifetime_peak\",\"prior_peak_bytes\":%llu,\"initial_current_bytes\":%llu,\"peak_bytes\":%llu,\"peak_rss_bytes\":%llu,\"seconds\":%.6f,\"exit_code\":%d,\"swap_configured\":false}\n",
        before,initial,after,(unsigned long long)usage.ru_maxrss*1024,seconds,code);
    return code;
}
