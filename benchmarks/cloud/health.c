#define _POSIX_C_SOURCE 200809L
/* Benchmark container liveness only. Not the transcription API. */
#include <arpa/inet.h>
#include <errno.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <sys/time.h>
#include <unistd.h>
int main(void) {
    const char *env=getenv("PORT");int port=env?atoi(env):8080;if(port<1||port>65535)return 2;
    signal(SIGPIPE,SIG_IGN);int fd=socket(AF_INET6,SOCK_STREAM,0),one=1,zero=0;
    if(fd<0)return 1;
    setsockopt(fd,SOL_SOCKET,SO_REUSEADDR,&one,sizeof(one));setsockopt(fd,IPPROTO_IPV6,IPV6_V6ONLY,&zero,sizeof(zero));
    struct sockaddr_in6 addr={0};addr.sin6_family=AF_INET6;addr.sin6_port=htons((uint16_t)port);
    if(bind(fd,(struct sockaddr *)&addr,sizeof(addr))||listen(fd,16)){perror("listen");return 1;}
    for(;;){int c=accept(fd,NULL,NULL);if(c<0){if(errno==EINTR)continue;return 1;}
        struct timeval timeout={2,0};setsockopt(c,SOL_SOCKET,SO_RCVTIMEO,&timeout,sizeof(timeout));setsockopt(c,SOL_SOCKET,SO_SNDTIMEO,&timeout,sizeof(timeout));
        char request[4096];ssize_t received=recv(c,request,sizeof(request)-1,0);
        if(received>0){request[received]=0;const char *body="{\"status\":\"ready\",\"role\":\"int8-benchmark\"}\n";
            char response[512];int n=snprintf(response,sizeof(response),"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: %zu\r\nConnection: close\r\n\r\n%s",strlen(body),body);
            for(int off=0;off<n;){ssize_t sent=send(c,response+off,(size_t)(n-off),0);if(sent<=0)break;off+=(int)sent;}}
        close(c);
    }
}
