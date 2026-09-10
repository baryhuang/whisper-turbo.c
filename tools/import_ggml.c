#define _POSIX_C_SOURCE 200809L
/* Streaming C importer for the published Whisper Turbo GGML F16/F32 model.
 * It writes the runtime's existing WHTRBO01 Q8 format; no Python conversion. */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <unistd.h>
#include "../src/generic/whisper_turbo_image.h"

#pragma pack(push,1)
struct header { char magic[8]; uint32_t version,count,layers,mels,state,heads,mlp,frames;
    uint64_t directory,data,bytes,reserved; };
struct desc { char name[96]; uint32_t kind,rank,shape[4]; uint64_t offset,bytes,reserved; };
#pragma pack(pop)
static struct desc descriptors[1024];
static uint32_t count;
static FILE *input,*output;
static void die(const char *s) { fprintf(stderr,"import: %s\n",s); exit(1); }
static void read_exact(void *p,size_t bytes) { if(fread(p,1,bytes,input)!=bytes)die("truncated input"); }
static void write_exact(const void *p,size_t bytes) {if(fwrite(p,1,bytes,output)!=bytes)die("write failed");}
static uint32_t u32(void) {uint32_t n;read_exact(&n,4);return n;}
static uint64_t align64(uint64_t n){return (n+63)&~UINT64_C(63);}
static float half_float(uint16_t h) {
    uint32_t sign=(uint32_t)(h&0x8000)<<16, e=(h>>10)&31, m=h&1023, bits;
    if(e==0) {
        if(!m)bits=sign;
        else {int exp=-14;while(!(m&1024)){m<<=1;--exp;}bits=sign|((uint32_t)(exp+127)<<23)|((m&1023)<<13);}
    } else bits=sign|((e==31?255:e+112)<<23)|(m<<13);
    float f;memcpy(&f,&bits,4);return f;
}
static struct desc *entry(const char *name,uint32_t kind,uint32_t rank,const uint32_t *shape) {
    if(count>=1024||strlen(name)>=96||rank>4)die("bad descriptor");
    for(uint32_t i=0;i<count;i++)if(!strcmp(name,descriptors[i].name))die("duplicate tensor");
    struct desc *d=&descriptors[count++];strcpy(d->name,name);d->kind=kind;d->rank=rank;
    memcpy(d->shape,shape,rank*4);d->offset=align64((uint64_t)ftello(output));
    if(fseeko(output,(off_t)d->offset,SEEK_SET))die("seek failed");
    return d;
}
static void finish_entry(struct desc *d){d->bytes=(uint64_t)ftello(output)-d->offset;}
static void payload(const char *name,uint32_t kind,uint32_t rank,const uint32_t *shape,const void *p,size_t bytes) {
    struct desc *d=entry(name,kind,rank,shape);write_exact(p,bytes);finish_entry(d);
}
static int map_name(const char *s,char *out,size_t n) {
    const char *simple[][2]={
      {"encoder.positional_embedding","encoder.embed_positions.weight"},
      {"decoder.positional_embedding","decoder.embed_positions.weight"},
      {"decoder.token_embedding.weight","decoder.embed_tokens.weight"},
      {"encoder.ln_post.weight","encoder.layer_norm.weight"},{"encoder.ln_post.bias","encoder.layer_norm.bias"},
      {"decoder.ln.weight","decoder.layer_norm.weight"},{"decoder.ln.bias","decoder.layer_norm.bias"}};
    for(size_t i=0;i<sizeof(simple)/sizeof(simple[0]);i++)if(!strcmp(s,simple[i][0])){
        snprintf(out,n,"%s",simple[i][1]);return 1;}
    if(!strncmp(s,"encoder.conv",12)){snprintf(out,n,"%s",s);return 1;}
    const char *side=!strncmp(s,"encoder.blocks.",15)?"encoder":!strncmp(s,"decoder.blocks.",15)?"decoder":NULL;
    if(!side)return 0;
    char *tail;long layer=strtol(s+15,&tail,10);if(layer<0||layer>=32||*tail!='.')return 0;tail++;
    const char *parts[][2]={{"attn_ln.","self_attn_layer_norm."},{"cross_attn_ln.","encoder_attn_layer_norm."},
        {"mlp_ln.","final_layer_norm."},{"attn.query.","self_attn.q_proj."},{"attn.key.","self_attn.k_proj."},
        {"attn.value.","self_attn.v_proj."},{"attn.out.","self_attn.out_proj."},
        {"cross_attn.query.","encoder_attn.q_proj."},{"cross_attn.key.","encoder_attn.k_proj."},
        {"cross_attn.value.","encoder_attn.v_proj."},{"cross_attn.out.","encoder_attn.out_proj."},
        {"mlp.0.","fc1."},{"mlp.2.","fc2."}};
    for(size_t i=0;i<sizeof(parts)/sizeof(parts[0]);i++) {
        size_t len=strlen(parts[i][0]);if(!strncmp(tail,parts[i][0],len)){
            snprintf(out,n,"%s.layers.%ld.%s%s",side,layer,parts[i][1],tail+len);return 1;}
    }return 0;
}
static void floats(float *v,size_t n,uint32_t type) {
    if(type==0)read_exact(v,n*4);
    else if(type==1){uint16_t h[1024];if(n>1024)die("internal block overflow");read_exact(h,n*2);for(size_t i=0;i<n;i++)v[i]=half_float(h[i]);}
    else die("only original F16/F32 GGML checkpoints are accepted");
    for(size_t i=0;i<n;i++)if(!isfinite(v[i]))die("nonfinite model weight");
}
int main(int argc,char **argv) {
    uint16_t endian=1;if(*(unsigned char *)&endian!=1)die("little-endian host required");
    if(argc!=3){fprintf(stderr,"usage: %s ggml-large-v3-turbo.bin OUTPUT.whtrbo\n",argv[0]);return 2;}
    input=fopen(argv[1],"rb");if(!input)die("cannot open input");
    /* Exclusive creation preserves an existing model image. */
    output=fopen(argv[2],"wbx");if(!output)die("cannot create output (must not exist)");
    if(u32()!=0x67676d6c)die("bad GGML magic");
    uint32_t hp[11];read_exact(hp,sizeof(hp));
    const uint32_t expected[10]={51866,1500,1280,20,32,448,1280,20,4,128};
    if(memcmp(hp,expected,sizeof(expected)))die("not Whisper large-v3-turbo");
    uint64_t data_offset=align64(sizeof(struct header)+sizeof(descriptors));
    if(fseeko(output,(off_t)data_offset,SEEK_SET))die("seek failed");
    uint32_t mel_shape[2]={u32(),u32()};
    if(mel_shape[0]!=128||mel_shape[1]!=201)die("bad mel dimensions");
    float mel[128*201];read_exact(mel,sizeof(mel));payload("frontend.mel_128",1,2,mel_shape,mel,sizeof(mel));
    uint32_t vocab=u32();if(vocab>51866||vocab<50257)die("bad vocab count");
    uint32_t offsets[51867]={0};unsigned char special[51866]={0};
    unsigned char *blob=malloc(4*1024*1024);if(!blob)die("allocation failed");size_t used=0;
    for(uint32_t i=0;i<51866;i++) {
        uint32_t len=i<vocab?u32():0;if(len>4096||used+len>4*1024*1024)die("vocabulary overflow");
        read_exact(blob+used,len);used+=len;offsets[i+1]=(uint32_t)used;special[i]=(i>=50257);
    }
    uint32_t shape=51867;payload("tokenizer.offsets",2,1,&shape,offsets,sizeof(offsets));
    shape=(uint32_t)used;payload("tokenizer.bytes",3,1,&shape,blob,used);free(blob);
    shape=51866;payload("tokenizer.special",3,1,&shape,special,sizeof(special));
    /* Pinned openai/whisper-large-v3-turbo generation_config.json suppress_tokens. */
    static const uint32_t suppress[]={1,2,7,8,9,10,14,25,26,27,28,29,31,58,59,60,61,62,63,90,91,92,93,
    359,503,522,542,873,893,902,918,922,931,1350,1853,1982,2460,2627,3246,3253,3268,3536,3846,3961,
    4183,4667,6585,6647,7273,9061,9383,10428,10929,11938,12033,12331,12562,13793,14157,14635,15265,
    15618,16553,16604,18362,18956,20075,21675,22520,26130,26161,26435,28279,29464,31650,32302,32470,
    36865,42863,47425,49870,50254,50258,50359,50360,50361,50362,50363};
    shape=sizeof(suppress)/sizeof(suppress[0]);payload("tokenizer.suppress",2,1,&shape,suppress,sizeof(suppress));
    for(;;) {
        uint32_t rank;size_t got=fread(&rank,1,4,input);if(!got)break;if(got!=4||rank<1||rank>3)die("bad tensor rank");
        uint32_t len=u32(),type=u32(),dims[4]={0},shape4[4]={0};
        if(len>=128||type>1)die("bad name length or tensor type");
        read_exact(dims,rank*4);
        uint64_t elements=1;for(uint32_t i=0;i<rank;i++){if(!dims[i]||dims[i]>100000)die("bad tensor shape");elements*=dims[i];shape4[rank-i-1]=dims[i];}
        char name[128]={0},mapped[96];read_exact(name,len);if(!map_name(name,mapped,sizeof(mapped)))die(name);
        if(strstr(mapped,".bias")&&elements==1280){rank=1;shape4[0]=1280;shape4[1]=shape4[2]=0;}
        int quant=rank==2&&shape4[1]%128==0&&(strstr(mapped,".layers.")||!strcmp(mapped,"decoder.embed_tokens.weight"));
        struct desc *d=entry(mapped,quant?5:1,rank,shape4);
        for(uint64_t pos=0;pos<elements;) {
            float v[1024];size_t n=quant?128:(elements-pos>1024?1024:(size_t)(elements-pos));floats(v,n,type);
            if(quant){float max=0;for(size_t i=0;i<n;i++)if(fabsf(v[i])>max)max=fabsf(v[i]);
                float scale=max/127.0f;uint32_t bits;memcpy(&bits,&scale,4);bits+=0x7fff+((bits>>16)&1);
                uint16_t bf=(uint16_t)(bits>>16);uint32_t rb=(uint32_t)bf<<16;memcpy(&scale,&rb,4);
                signed char q[128];for(size_t i=0;i<128;i++){long z=scale?lrintf(v[i]/scale):0;if(z>127)z=127;if(z< -127)z=-127;q[i]=(signed char)z;}
                write_exact(&bf,2);write_exact(q,128);
            }else write_exact(v,n*4);pos+=n;
        }finish_entry(d);
    }
    uint64_t bytes=align64((uint64_t)ftello(output));if(fflush(output)||ftruncate(fileno(output),(off_t)bytes))die("finalize failed");
    struct header h={"WHTRBO01",3,count,32,128,1280,20,5120,1500,sizeof(struct header),data_offset,bytes,0};
    rewind(output);write_exact(&h,sizeof(h));write_exact(descriptors,count*sizeof(descriptors[0]));
    if(fclose(output))die("close failed");
    fclose(input);
    cllm_whisper_turbo_model model;if(cllm_whisper_turbo_model_open(argv[2],&model))die("runtime rejected imported image");
    cllm_whisper_turbo_model_close(&model);
    printf("{\"format\":\"WHTRBO01\",\"precision\":\"q8\",\"descriptors\":%u,\"bytes\":%llu}\n",count,(unsigned long long)bytes);
    return 0;
}
