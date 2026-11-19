/* Experiment-only bridge, linked to the patched FFmpeg 7.1.1 static libraries.
 * Single thread, progressive 8-bit 4:2:0. stdout/stdin are binary IPC with Python.
 * Modified references are the decoder's own AVBuffer-backed DPB pictures.
 */
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <fcntl.h>
#include <io.h>
#include "libavcodec/avcodec.h"
#include "libavcodec/h264dec.h"
#include "libavcodec/h264_parse.h"
#include "libavformat/avformat.h"
#include "libavutil/cpu.h"
#include "libavutil/time.h"

static int inject, seq, seen[6], active;
static int32_t *export_grid;
static int export_w, export_h, export_seq = -1;
static int enabled;
static FILE *trace, *motion;
static uint64_t coeff_hash = 1469598103934665603ULL;
static int counts[6];
static const H264Context *current_h;
static int mb_x, mb_y, sample_id;
static int last_type;
static int64_t hook_us, decode_i_us, decode_pb_us, decode_other_us;
static int64_t init_us,demux_us,receive_us,output_ipc_us;

static void fail(const char *message) { fprintf(stderr, "%s\n", message); exit(2); }
static void exact_read(void *p, size_t n) {
    if (fread(p, 1, n, stdin) != n) fail("Python reference response truncated");
}
static void header(int event, AVFrame *f, int poc, int fn, int nal, int reference) {
    int32_t v[8] = {event, f->width-(int)f->crop_left-(int)f->crop_right,
        f->height-(int)f->crop_top-(int)f->crop_bottom, poc, f->pict_type, fn, nal, reference};
    fwrite(v, sizeof(v), 1, stdout); fwrite(&f->pts, 8, 1, stdout);
}
static void planes(AVFrame *f, int replace) {
    int w=f->width-f->crop_left-f->crop_right, h=f->height-f->crop_top-f->crop_bottom;
    for (int p=0;p<3;p++) {
        int shift=p!=0, pw=w>>shift, ph=h>>shift;
        int left=f->crop_left>>shift, top=f->crop_top>>shift;
        uint8_t *base=f->data[p]+top*f->linesize[p]+left;
        for(int y=0;y<ph;y++) {
            uint8_t *row=base+y*f->linesize[p];
            if(replace) exact_read(row,pw); else fwrite(row,1,pw,stdout);
            if(replace) {
                memset(row-left,row[0],left);
                memset(row+pw,row[pw-1],(f->width>>shift)-left-pw);
            }
        }
        if(replace) {
            for(int y=0;y<top;y++) memcpy(f->data[p]+y*f->linesize[p],base-left,f->width>>shift);
            for(int y=top+ph;y<(f->height>>shift);y++)
                memcpy(f->data[p]+y*f->linesize[p],base+(ph-1)*f->linesize[p]-left,f->width>>shift);
        }
    }
}

void ff_stage3_picture(H264Context *h) {
    if(!enabled) return;
    int64_t hook_start=av_gettime_relative();
    AVFrame *f=h->cur_pic_ptr->f;
    last_type=f->pict_type;
    if(f->format!=AV_PIX_FMT_YUV420P || h->picture_structure!=PICT_FRAME || FRAME_MBAFF(h))
        fail("Unsupported stream: progressive 8-bit YUV420P required");
    if(trace) {
        fprintf(trace,"{\"event\":\"frame\",\"seq\":%d,\"pts\":%lld,\"poc\":%d,\"frame_num\":%d,\"type\":%d,\"nal\":%d,\"reference\":%d,\"coeff_hash\":\"%016llx\",\"mb_categories\":[%d,%d,%d,%d,%d,%d]}\n",
            seq,(long long)f->pts,h->cur_pic_ptr->poc,h->poc.frame_num,f->pict_type,h->nal_unit_type,
            h->cur_pic_ptr->reference,(unsigned long long)coeff_hash,counts[0],counts[1],counts[2],counts[3],counts[4],counts[5]);
    }
    if(inject == 1 && f->pict_type==AV_PICTURE_TYPE_I) {
        /* Preserve hidden coded rows during identity roundtrips. ROP controls their
         * replacement explicitly; a display-size copy is insufficient for the DPB. */
        AVFrame coded=*f;
        coded.crop_left=coded.crop_right=coded.crop_top=coded.crop_bottom=0;
        header(1,&coded,h->cur_pic_ptr->poc,h->poc.frame_num,h->nal_unit_type,h->cur_pic_ptr->reference);
        planes(&coded,0); fflush(stdout); planes(&coded,1);
    }
    if(inject == 2) {
        /* AC bridge: preserve visible pixels and export a ceiling-sized grid.
         * Python locally refreshes incomplete edge cells instead of inventing motion. */
        if(f->crop_left || f->crop_top) fail("Export requires zero left/top crop");
        int visible_w = f->width-f->crop_right, visible_h = f->height-f->crop_bottom;
        int grid_w = (visible_w+7)/8, grid_h = (visible_h+7)/8;
        if(export_w < grid_w || export_h < grid_h) fail("Export grid mismatch");
        header(3,f,h->cur_pic_ptr->poc,h->poc.frame_num,h->nal_unit_type,h->cur_pic_ptr->reference);
        planes(f,0);
        for(int list=0;list<2;list++) {
            for(int row=0;row<grid_h;row++) {
                fwrite(export_grid+((list*export_h+row)*export_w)*4,
                    sizeof(int32_t),grid_w*4,stdout);
            }
        }
        fflush(stdout);
    }
    seq++; memset(seen,0,sizeof(seen)); memset(counts,0,sizeof(counts));
    coeff_hash=1469598103934665603ULL;
    hook_us+=av_gettime_relative()-hook_start;
}

void ff_stage3_mb_begin(const H264Context *h, H264SliceContext *sl) {
    active=0; if(!enabled || (!trace && inject != 2)) return;
    int type=h->cur_pic.mb_type[sl->mb_xy];
    if(inject == 2) {
        int gw=h->mb_width*2, gh=h->mb_height*2;
        if(export_seq!=seq) {
            if(gw!=export_w || gh!=export_h) {
                free(export_grid); export_grid=malloc((size_t)gw*gh*2*4*sizeof(int32_t));
                if(!export_grid) fail("Export allocation failed");
                export_w=gw; export_h=gh;
            }
            for(int i=0;i<gw*gh*2*4;i++) export_grid[i]=INT_MIN;
            export_seq=seq;
        }
        for(int list=0;list<2;list++) for(int by=0;by<2;by++) for(int bx=0;bx<2;bx++) {
            int k=by*8+bx*4, index=scan8[k], r=sl->ref_cache[list][index];
            if(IS_INTRA(type) || !USES_LIST(type,list) || list>=sl->list_count || r<0 || r>=sl->ref_count[list]) continue;
            int32_t *record=export_grid+((list*gh+sl->mb_y*2+by)*gw+sl->mb_x*2+bx)*4;
            record[0]=r; record[1]=sl->mv_cache[list][index][0]; record[2]=sl->mv_cache[list][index][1]; record[3]=sl->ref_list[list][r].poc;
        }
    }
    if(!trace) return;
    int cat=IS_INTRA(type)?0:(IS_SKIP(type)?4:(USES_LIST(type,0)?(USES_LIST(type,1)?3:1):2));
    counts[cat]++;
    const uint8_t *bytes=(const uint8_t *)sl->mb;
    for(int k=0;k<sizeof(sl->mb);k++) { coeff_hash^=bytes[k]; coeff_hash*=1099511628211ULL; }
    /* Every macroblock's modes, two-list refs and quarter-pel vectors, before IDCT.
     * Binary int32 records: seq,x,y,type,qscale,cbp, then 2*16*(ref,mvx,mvy,poc).
     */
    if(motion) {
        int32_t record[134]={seq,sl->mb_x,sl->mb_y,type,sl->qscale,sl->cbp}; int n=6;
        for(int list=0;list<2;list++) for(int k=0;k<16;k++) {
            int index=scan8[k], r=sl->ref_cache[list][index];
            record[n++]=r; record[n++]=sl->mv_cache[list][index][0]; record[n++]=sl->mv_cache[list][index][1];
            record[n++]=(!IS_INTRA(type)&&list<sl->list_count&&r>=0&&r<sl->ref_count[list])?sl->ref_list[list][r].poc:INT_MIN;
        }
        fwrite(record,sizeof(record),1,motion);
    }
    /* One first example per category per picture; full unmodified coefficients. */
    if(seen[cat]++) return;
    active=1; current_h=h; mb_x=sl->mb_x; mb_y=sl->mb_y; sample_id=cat;
    fprintf(trace,"{\"event\":\"mb\",\"seq\":%d,\"x\":%d,\"y\":%d,\"category\":%d,\"mb_type\":%d,\"qscale\":%d,\"cbp\":%d,\"transform8\":%d,\"refs\":[",seq,mb_x,mb_y,cat,type,sl->qscale,sl->cbp,!!IS_8x8DCT(type));
    for(int list=0;list<2;list++) {
        if(list)fputc(',',trace); fputc('[',trace);
        for(int k=0;k<sl->ref_count[list];k++) fprintf(trace,"%s%d",k?",":"",sl->ref_list[list][k].poc);
        fputc(']',trace);
    }
    fprintf(trace,"],\"coefficients\":[");
    for(int k=0;k<sizeof(sl->mb)/sizeof(sl->mb[0]);k++) fprintf(trace,"%s%d",k?",":"",sl->mb[k]);
    fprintf(trace,"]}\n");
}
void ff_stage3_mb_end(void) { active=0; }

int ff_stage3_add(uint8_t *address, int prediction, int residual, int maximum) {
    int value=prediction+residual; value=value<0?0:value>maximum?maximum:value;
    if(active) {
        const AVFrame *f=current_h->cur_pic.f;
        for(int p=0;p<3;p++) {
            intptr_t delta=(intptr_t)address-(intptr_t)f->data[p];
            if(delta>=0 && delta<(intptr_t)f->linesize[p]*f->height/(p?2:1)) {
                int x=delta%f->linesize[p],y=delta/f->linesize[p];
                fprintf(trace,"{\"event\":\"residual\",\"seq\":%d,\"category\":%d,\"plane\":%d,\"x\":%d,\"y\":%d,\"prediction\":%d,\"residual\":%d,\"reconstructed\":%d}\n",seq,sample_id,p,x,y,prediction,residual,value);
                break;
            }
        }
    }
    return value;
}

static void receive(AVCodecContext *c) {
    AVFrame *f=av_frame_alloc(); int r;
    for(;;) {
        int64_t tick=av_gettime_relative();
        r=avcodec_receive_frame(c,f);receive_us+=av_gettime_relative()-tick;
        if(r<0)break;
        tick=av_gettime_relative();
        if(inject != 2) { header(2,f,0,0,0,0); planes(f,0); fflush(stdout); }
        av_frame_unref(f);
        output_ipc_us+=av_gettime_relative()-tick;
    }
    av_frame_free(&f);
    if(r!=AVERROR(EAGAIN)&&r!=AVERROR_EOF) fail("receive_frame failed");
}
int main(int argc,char **argv) {
    int64_t startup=av_gettime_relative();
    if(argc<3) fail("native_decoder input.mp4 inject(0/1) [trace.jsonl] [motion.bin]");
    _setmode(_fileno(stdin),_O_BINARY); _setmode(_fileno(stdout),_O_BINARY);
    setvbuf(stdout, NULL, _IOFBF, 1 << 20);
    av_log_set_level(AV_LOG_ERROR);
    inject=atoi(argv[2]); if(argc>3) trace=fopen(argv[3],"wb"); if(argc>4) motion=fopen(argv[4],"wb");
    AVFormatContext *fmt=NULL;
    if(avformat_open_input(&fmt,argv[1],NULL,NULL)<0 || avformat_find_stream_info(fmt,NULL)<0) fail("input open failed");
    int v=av_find_best_stream(fmt,AVMEDIA_TYPE_VIDEO,-1,-1,NULL,0);
    if(v<0 || fmt->streams[v]->codecpar->codec_id!=AV_CODEC_ID_H264) fail("H.264 required");
    AVCodecContext *c=avcodec_alloc_context3(avcodec_find_decoder(AV_CODEC_ID_H264));
    avcodec_parameters_to_context(c,fmt->streams[v]->codecpar); c->thread_count=1; c->thread_type=0;
    c->pkt_timebase=fmt->streams[v]->time_base; c->err_recognition=AV_EF_EXPLODE;
    if(avcodec_open2(c,c->codec,NULL)<0) fail("codec open failed");
    /* avformat_find_stream_info may privately decode probe pictures. These must
     * never trigger ROP or pollute reconstruction evidence. */
    enabled=1;
    init_us=av_gettime_relative()-startup;
    const char *gop_env=getenv("STAGE3_GOP");
    int gop=gop_env?atoi(gop_env):-1, submitted=0;
    if(gop>=0 && av_seek_frame(fmt,v,(int64_t)gop*15360,AVSEEK_FLAG_BACKWARD)<0) fail("GOP seek failed");
    AVPacket *pkt=av_packet_alloc();
    for(;;) {
        int64_t tick=av_gettime_relative();int read_result=av_read_frame(fmt,pkt);
        demux_us+=av_gettime_relative()-tick;
        if(read_result<0)break;
        if(pkt->stream_index==v) {
            if(gop>=0 && submitted && (pkt->flags&AV_PKT_FLAG_KEY)) { av_packet_unref(pkt); break; }
            if(gop>=0 && !submitted && (!(pkt->flags&AV_PKT_FLAG_KEY) || pkt->pts!=(int64_t)gop*15360)) fail("Seek did not reach requested IDR");
            submitted++;
            int64_t tick=av_gettime_relative(), before=hook_us;
            last_type=0;
            if(avcodec_send_packet(c,pkt)<0) fail("send_packet failed");
            int64_t elapsed=av_gettime_relative()-tick-(hook_us-before);
            if(last_type==AV_PICTURE_TYPE_I) decode_i_us+=elapsed;
            else if(last_type) decode_pb_us+=elapsed; else decode_other_us+=elapsed;
            receive(c);
        }
        av_packet_unref(pkt);
    }
    avcodec_send_packet(c,NULL); receive(c);
    av_packet_free(&pkt); avcodec_free_context(&c); avformat_close_input(&fmt);
    if(trace)fclose(trace); if(motion)fclose(motion);
    fprintf(stderr,"{\"i_decode_seconds\":%.6f,\"pb_decode_reconstruction_seconds\":%.6f,\"other_decode_seconds\":%.6f,\"hook_seconds\":%.6f,\"pictures\":%d,\"native_initialization_probe_seconds\":%.6f,\"native_demux_seconds\":%.6f,\"native_receive_api_seconds\":%.6f,\"native_output_ipc_wait_seconds\":%.6f}\n",decode_i_us/1e6,decode_pb_us/1e6,decode_other_us/1e6,hook_us/1e6,seq,init_us/1e6,demux_us/1e6,receive_us/1e6,output_ipc_us/1e6);
    free(export_grid);
    return 0;
}
