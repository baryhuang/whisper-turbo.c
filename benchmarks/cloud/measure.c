#define _GNU_SOURCE
/* Dedicated Linux cgroup-v2 measurement, including descendants and file cache.
 * Runs only on a disposable benchmark guest with delegated cgroup control.
 * Usage: measure LIMIT_BYTES MODEL_TO_EVICT COMMAND [ARG...]
 * LIMIT_BYTES=0 means unlimited. Swap is always disabled for the child. */
#include <errno.h>
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/resource.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <time.h>
#include <unistd.h>

static int put(const char *path,const char *value) {
    int fd=open(path,O_WRONLY|O_CLOEXEC);
    if(fd<0){perror(path);return -1;}
    size_t n=strlen(value);ssize_t written=write(fd,value,n);
    int saved=errno;close(fd);errno=saved;
    if(written!=(ssize_t)n){perror(path);return -1;}return 0;
}
static unsigned long long get(const char *path) {
    unsigned long long n=0;FILE *f=fopen(path,"r");
    if(!f||fscanf(f,"%llu",&n)!=1){perror(path);exit(2);}
    fclose(f);return n;
}
static double now(void){struct timespec t;clock_gettime(CLOCK_MONOTONIC,&t);return t.tv_sec+t.tv_nsec*1e-9;}
int main(int argc,char **argv) {
    if(argc<4){fprintf(stderr,"usage: %s LIMIT_BYTES MODEL COMMAND [ARG...]\n",argv[0]);return 2;}
    char *end;errno=0;unsigned long long limit=strtoull(argv[1],&end,10);
    if(errno||*end||!argv[1][0]||argv[1][0]=='-')return 2;
    int fd=open(argv[2],O_RDONLY|O_CLOEXEC);
    if(fd<0){perror(argv[2]);return 2;}
    int advised=posix_fadvise(fd,0,0,POSIX_FADV_DONTNEED);close(fd);
    if(advised){errno=advised;perror("evict model cache");return 2;}
    if(put("/sys/fs/cgroup/cgroup.subtree_control","+memory"))return 2;
    char group[160],path[200],value[80];
    snprintf(group,sizeof(group),"/sys/fs/cgroup/whisper-bench-%ld",(long)getpid());
    if(mkdir(group,0755)){perror(group);return 2;}
    snprintf(path,sizeof(path),"%s/memory.max",group);
    if(limit)snprintf(value,sizeof(value),"%llu",limit);else strcpy(value,"max");
    if(put(path,value))return 2;
    snprintf(path,sizeof(path),"%s/memory.swap.max",group);if(put(path,"0"))return 2;
    double start=now();pid_t child=fork();
    if(child<0){perror("fork");return 2;}
    if(child==0) {
        snprintf(path,sizeof(path),"%s/cgroup.procs",group);
        if(put(path,"0"))_exit(125);
        execvp(argv[3],argv+3);perror("exec");_exit(127);
    }
    int status;struct rusage usage;
    while(wait4(child,&status,0,&usage)<0)if(errno!=EINTR){perror("wait4");return 2;}
    double elapsed=now()-start;
    snprintf(path,sizeof(path),"%s/memory.peak",group);unsigned long long peak=get(path);
    snprintf(path,sizeof(path),"%s/memory.events",group);
    FILE *events=fopen(path,"r");char key[64];unsigned long long val,oom=0,kills=0;
    if(!events){perror(path);return 2;}
    while(fscanf(events,"%63s %llu",key,&val)==2){if(!strcmp(key,"oom"))oom=val;if(!strcmp(key,"oom_kill"))kills=val;}
    fclose(events);
    int code=WIFEXITED(status)?WEXITSTATUS(status):128+WTERMSIG(status);
    fprintf(stderr,"MEASUREMENT {\"limit_bytes\":%llu,\"cgroup_peak_bytes\":%llu,\"peak_rss_bytes\":%llu,\"seconds\":%.6f,\"exit_code\":%d,\"oom_events\":%llu,\"oom_kills\":%llu,\"swap_disabled\":true}\n",
        limit,peak,(unsigned long long)usage.ru_maxrss*1024,elapsed,code,oom,kills);
    /* Only our empty per-run cgroup is removed; never a shared hierarchy. */
    if(rmdir(group)){perror("remove measurement cgroup");return 2;}
    return code;
}
