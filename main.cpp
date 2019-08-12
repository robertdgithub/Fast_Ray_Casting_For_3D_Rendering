
#include <vector>
#include <cmath>
#include <cstdlib>
#include <cstdio>
#undef min
#undef max
#define M_PI 3.14159265358979323846264338328f
float inf = 1e10;

inline float f_min(float a, float b) { return a<b ? a : b; }
inline float f_max(float a, float b) { return a>b ? a : b; }
float saturate(float x){ return x<0 ? 0 : x>1 ? 1 : x; }/* saturate���ͣ�λ��0��1֮��*/
unsigned char to_char(float x)
{
    return (unsigned char)(saturate(0.8f*powf(x,1/2.2f))*255);/*255*0.8*x��1/2.2�η�*/
}

struct vec
{
    float x, y, z;
    vec():x(0),y(0),z(0){}
    vec(float a_):x(a_),y(a_),z(a_){}
    vec(float x_, float y_, float z_):x(x_),y(y_),z(z_){}
};

vec operator+ (vec& a, float b) { return vec(a.x+b, a.y+b, a.z+b); }
vec operator+ (vec& a, vec& b) { return vec(a.x+b.x, a.y+b.y, a.z+b.z); }
vec operator- (vec& a, vec& b) { return vec(a.x-b.x, a.y-b.y, a.z-b.z); }
vec operator* (vec& a, float b) { return vec(a.x*b, a.y*b, a.z*b); }
vec normalize(vec& a) { float len = sqrtf(a.x*a.x+a.y*a.y+a.z*a.z); return a*(1.0f/len); }
inline float dot(vec& a, vec& b) { return a.x*b.x+a.y*b.y+a.z*b.z; }
inline vec mult(vec& a, vec& b) { return vec(a.x*b.x, a.y*b.y, a.z*b.z); }
vec cross(vec& a, vec& b) { return vec(a.y*b.z-a.z*b.y,a.z*b.x-a.x*b.z,a.x*b.y-a.y*b.x); }
float dist_sq(vec& a, vec& b) { vec d(a-b); return dot(d,d); }
float dist(vec& a, vec& b) { return sqrtf(dist_sq(a,b)); }
float length(vec& a) { return sqrtf(dot(a,a)); }
vec expf(vec& a) { return vec(expf(a.x),expf(a.y),expf(a.z)); }
inline vec f_min(vec& a, float b) { return vec(f_min(a.x,b),f_min(a.y,b),f_min(a.z,b)); }
inline vec f_max(vec& a, float b) { return vec(f_max(a.x,b),f_max(a.y,b),f_max(a.z,b)); }

struct ray
{
    vec o, d;
    ray(vec& o_, vec& d_):o(o_),d(normalize(d_)){}
    vec advance(float t) { return o + d * t; }
};

struct triangle
{
    vec a, b, c;
    triangle(){}
    triangle(vec& a_, vec& b_, vec& c_):a(a_),b(b_),c(c_){}
    bool intersect(ray& r, float& t) /*�����󽻣��Ƕȷ�*/
        {
        vec e1 = b - a;
        vec e2 = c - a;
        vec pvec = cross(r.d, e2);
        float det = dot(e1, pvec);
        if (det == 0) return false;
        float invDet = 1 / det;
        vec tvec = r.o - a;
        float u = dot(tvec, pvec) * invDet;
        if (u < 0 || u > 1) return false;
        vec qvec = cross(tvec, e1);
        float v = dot(r.d, qvec) * invDet;
        if (v < 0 || u + v > 1) return false;
        t = dot(e2, qvec) * invDet;
        return true;
    }
};

float signed_map(int x, int n){ return 2*(x/(float)n)-1; }
float sq(float x){ return x*x; }

float implicit(int i, int j, int k, int N)    /*������ά���ݣ���sampler��i,j,kΪ��0��ʼ�ĵ�����*/
{
    float x = signed_map(i,N) * 1.5f; /*  ((2����i/N)-1)*1.5  */
    float y = signed_map(j,N) * 1.5f;
    float z = signed_map(k,N) * 1.5f;
    return
    sq(2.92f*(x-1)*sq(x)*(x+1)+1.7f*sq(y))*sq(sq(y)-0.88f)+
    sq(2.92f*(y-1)*sq(y)*(y+1)+1.7f*sq(z))*sq(sq(z)-0.88f)+
    sq(2.92f*(z-1)*sq(z)*(z+1)+1.7f*sq(x))*sq(sq(x)-0.88f)-0.02f -0.03f;
}

typedef float (*implicit_function)(int i, int j, int k, int N);
implicit_function sampler = &implicit;

typedef int (*index_convention)(int, int, int, int, int);/*����ת��*/
inline int index_xyz(int x, int y, int z, int N, int NN) { return x+y*N+z*NN; }
inline int index_xzy(int x, int y, int z, int N, int NN) { return x+z*N+y*NN; }
inline int index_yxz(int x, int y, int z, int N, int NN) { return y+x*N+z*NN; }
inline int index_yzx(int x, int y, int z, int N, int NN) { return y+z*N+x*NN; }
inline int index_zxy(int x, int y, int z, int N, int NN) { return z+x*N+y*NN; }
inline int index_zyx(int x, int y, int z, int N, int NN) { return z+y*N+x*NN; }

index_convention g_index = &index_xyz;

struct tex3d
{
    int N, total, NN;
    vec min, max;
    float l;
    std::vector<float> data;
    vec vert[8]; /* 8������ */
    triangle tri[12];
    inline int index(int x, int y, int z)
        {
        return g_index(x, y, z, N, NN);
    }
    vec get_voxel_position(int i, int j, int k)
        {
        return vec((i+0.5f)/(float)N,(j+0.5f)/(float)N,(k+0.5f)/(float)N)  *l + min;
    }
    float& operator[] (int n) { return data[n]; }
    void init_implicit()  /*������ά����*/
        {
        for(int i=0; i<N; i++)
                {
            for(int j=0; j<N; j++)
                        {
                for(int k=0; k<N; k++)
                                {
                    data[index(i,j,k)] =
                        0.1f*powf(saturate((-sampler(i,j,k,N))+0.99f),50.0f); /* sampler����implicit */
                }
            }
        }
    }
    tex3d(int N_=100, float l_=1, vec& ref=vec(-0.5f))
                :N(N_), NN(N_*N_), total(N_*N_*N_), l(l_), min(ref), max(ref+l_)/*��ʼ���б���
        N��NN�Ƚ��г�ʼ��       һ��������8�����㣬12������*/
    {
        data.resize(total);
        data.assign(total, 0.5);
        vert[0] = vec(0,0,0)*l + min;
        vert[1] = vec(0,0,1)*l + min;
        vert[2] = vec(0,1,0)*l + min;
        vert[3] = vec(0,1,1)*l + min;
        vert[4] = vec(1,0,0)*l + min;
        vert[5] = vec(1,0,1)*l + min;
        vert[6] = vec(1,1,0)*l + min;
        vert[7] = vec(1,1,1)*l + min;
        tri[ 0] = triangle(vert[0],vert[1],vert[3]);
        tri[ 1] = triangle(vert[0],vert[3],vert[2]);
        tri[ 2] = triangle(vert[4],vert[6],vert[7]);
        tri[ 3] = triangle(vert[4],vert[7],vert[5]);
        tri[ 4] = triangle(vert[0],vert[5],vert[1]);
        tri[ 5] = triangle(vert[0],vert[4],vert[5]);
        tri[ 6] = triangle(vert[2],vert[3],vert[7]);
        tri[ 7] = triangle(vert[2],vert[7],vert[6]);
        tri[ 8] = triangle(vert[0],vert[2],vert[6]);
        tri[ 9] = triangle(vert[0],vert[6],vert[4]);
        tri[10] = triangle(vert[1],vert[5],vert[7]);
        tri[11] = triangle(vert[1],vert[7],vert[3]);
    }
    vec remap_to_one(vec& pos)
        {
        return (pos-min)*(1.0f/l);
    }
    float fetch(vec& pos) /*�����Բ�ֵ�ز���*/
        {
        if(outside(pos))return 0;
        vec p = remap_to_one(pos);
        float x_ = p.x * (N-2);
        float y_ = p.y * (N-2);
        float z_ = p.z * (N-2);
        int i_0 = (int)x_; int i_1 = i_0 + 1;
        int j_0 = (int)y_; int j_1 = j_0 + 1;
        int k_0 = (int)z_; int k_1 = k_0 + 1;
        float u1 = x_ - i_0; float u0 = 1 - u1;
        float v1 = y_ - j_0; float v0 = 1 - v1;
        float w1 = z_ - k_0; float w0 = 1 - w1;
        return u0 *(v0*(w0*(data[index(i_0,j_0,k_0)])
                       +w1*(data[index(i_0,j_0,k_1)]))
                +   v1*(w0*(data[index(i_0,j_1,k_0)])
                       +w1*(data[index(i_0,j_1,k_1)])))
             + u1 *(v0*(w0*(data[index(i_1,j_0,k_0)])
                       +w1*(data[index(i_1,j_0,k_1)]))
                +   v1*(w0*(data[index(i_1,j_1,k_0)])
                       +w1*(data[index(i_1,j_1,k_1)])));
    }
    bool outside(vec& pos)
        {
        return pos.x<min.x || pos.y<min.y || pos.z<min.z
            || pos.x>max.x || pos.y>max.y || pos.z>max.z;
    }
};

bool intersect(ray& r, tex3d& s, float& t_near, float& t_far) /*����*/
{
    t_near = inf;
    t_far = -inf;
    bool any = false;
    float t;
    for(int n=0; n<12; n++)
        {
        bool hit = s.tri[n].intersect(r, t);
        if(hit && t<t_near) t_near = t;
        if(hit && t>t_far) t_far = t;
        any |= hit;
    }
    return any && t_near>0;
}


float g_sigma_t = 150;
float g_light_emission = 0.4f;
float ambient_emission = 0;

#define STEP 0.01f
vec g_volume_color(vec(1)-vec(0.10f,0.06f,0.02f));

vec lightcast(vec& voxel_pos, vec& light_pos, tex3d& density_volume, float light_emission)  /*����Ͷ��*/
{
    ray light_ray(light_pos,normalize(voxel_pos-light_pos)); /*�������㣬����*/
    float t_near, t_far;
    if(!intersect(light_ray, density_volume, t_near, t_far)) return vec(0); /*�������ཻ*/
    vec front = light_ray.advance(t_near);/*ǰ��*/
    vec back = light_ray.advance(t_far);/*����*/
    vec dir = light_ray.d;/*Ͷ�䷽��*/
    vec emission(light_emission);/*Ͷ��*/
    vec step = dir * STEP; /*Ͷ��ǰ��   #define STEP 0.01f  */
    float max_t = dist_sq(voxel_pos,light_pos); /* ����Ͷ�䳤�� */
    vec pos = front;
    do{
                float density = density_volume.fetch(pos); /* �ز����ĵ� */
        emission = mult(emission, expf(g_volume_color*(-g_sigma_t*density*length(step))));
        pos = pos + step;/* ����ǰ��һ�� */
    }while(dist_sq(pos,light_pos)<=max_t);/* ѭ��ֱ����������*/
    return f_max(emission, ambient_emission) * saturate(density_volume.fetch(voxel_pos)/0.6f);/* �Ƚϴ�С */
}        /* �Ƚϴ�С������һ����  float ambient_emission = 0;*/

void generate_light_volume(vec& light_pos, tex3d light_volume[3], tex3d& density_volume, float light_emission) /*������ά������*/
{
    int N = light_volume[0].N;
#pragma omp parallel for/*�� for ѭ�����л�����*/
    for(int i=0; i<N; i++)
        {
        for(int j=0; j<N; j++)
                {
            for(int k=0; k<N; k++)
                        {
                int n = light_volume[0].index(i,j,k);
                vec voxel_pos = light_volume[0].get_voxel_position(i,j,k);
                vec emission = lightcast(voxel_pos, light_pos, density_volume, light_emission);
                light_volume[0][n] = emission.x;
                light_volume[1][n] = emission.y;
                light_volume[2][n] = emission.z;
            }
        }
    }
    printf("generated light scatter volume\n"); /*������ά������*/
}

vec raycast_2nd_pass(ray& cam_ray, tex3d& density_volume, tex3d light_volume[3]) /*����Ͷ���ڶ��׶�*/
{
    float t_near, t_far;
    if(!intersect(cam_ray, density_volume, t_near, t_far)) return vec(0);
    vec front = cam_ray.advance(t_near); /* o+t_near * d */
    vec back = cam_ray.advance(t_far);  /*  o+t_far * d */
    vec dir = cam_ray.d;  /*  ����  */
    vec radiance(0);
    vec attenuation(1);
    vec step = dir * STEP; /*  #define STEP 0.01f   */
    vec pos = front; /* ��ǰ���� */
    do{
                float density = density_volume.fetch(pos);
        vec scatter(light_volume[0].fetch(pos),light_volume[1].fetch(pos),light_volume[2].fetch(pos));
        radiance = radiance + mult(scatter, attenuation);
        attenuation = mult(attenuation, expf(g_volume_color*(-g_sigma_t*density*length(step))));
        pos = pos + step;
    }while(!density_volume.outside(pos));
    return vec(radiance);
}


#define NOT(x) (!(x))
#define STATIC_LIGHT 0

int main(){
    int iw(300), ih(300);/*����*/
    std::vector<unsigned char> frame;
    frame.resize(iw*ih*3);

    tex3d vol(200,1,vec(-0.5));/*tex3d(int N_=100, float l_=1, vec& ref=vec(-0.5f))*/
    vol.init_implicit();
#if STATIC_LIGHT
    vec light_pos(2,0,0);/*  ����  vec(float x_, float y_, float z_):x(x_),y(y_),z(z_){}*/
    tex3d light_volume[3];/*light_volume������3��tex3dԪ��*/
    generate_light_volume(light_pos, light_volume, vol, g_light_emission);
#endif
    float dist = 2;
    float fov = 45;

    int n = 0;
    for(float x_rot=0; x_rot<M_PI*2; x_rot += 0.1f)
        {
        vec lookat(0,0,0);
        vec up(0,1,0);
        vec cam_o(dist*sinf(x_rot),0.2f,dist*cosf(x_rot));
        vec cam_d(normalize(lookat-cam_o));
        vec cam_x(normalize(cross(cam_d,up)));
        vec cam_y(cross(cam_x,cam_d));
#if NOT(STATIC_LIGHT)
        vec light_pos(lookat+cam_x*2);
        tex3d light_volume[3];
        generate_light_volume(light_pos, light_volume, vol, g_light_emission);/* ���ɱ��������� */
#endif
        for(int j=ih-1; j>=0; j--)  /*����һ��ppm д���ļ���*/
                {
#pragma omp parallel for
            for(int i=0; i<iw; i++)
                        {
                ray r(cam_o, cam_d
                    +cam_x*(signed_map(i,iw)*tan(fov*0.5f/180.0f*M_PI))
                    +cam_y*(signed_map(j,ih)*tan(fov*0.5f/180.0f*M_PI)));
                vec c = raycast_2nd_pass(r, vol, light_volume);
                int offset = (i+(ih-1-j)*iw)*3;
                frame[offset  ] = to_char(c.x);
                frame[offset+1] = to_char(c.y);
                frame[offset+2] = to_char(c.z);
            }
        }

        char fn[256];
        sprintf(fn,"%05d.ppm",n++);
        printf("%s\n",fn);
        FILE *fp = fopen(fn,"wb");
        fprintf(fp,"P6\n%d %d\n255\n",iw,ih);
        fwrite(&frame[0],1,frame.size(),fp);
        fclose(fp);
    }
    return 0;
}
