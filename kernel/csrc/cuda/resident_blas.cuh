// Experimental strict-FP64 device linear algebra for bounded-simplex SLSQP.
// Not integrated or qualified. LAPACK routines translated from OpenBLAS 0.3.30
// commit 993fad6aebbce34a97d3f8c34d6d79d35b64cc48, lapack-netlib/SRC/*.f.
// The bundled generated *.c files are not equivalent to those Fortran sources.
// This serial policy runs on the resident controller's owner thread; reductions
// are explicit and are NOT claimed bitwise equivalent to architecture-specific
// OpenBLAS. Source licenses and acquisition evidence accompany the port.
// Copyright (c) 1992-2023 The University of Tennessee and The University
//                         of Tennessee Research Foundation.  All rights
//                         reserved.
// Copyright (c) 2000-2023 The University of California Berkeley. All
//                         rights reserved.
// Copyright (c) 2006-2023 The University of Colorado Denver.  All rights
//                         reserved.
//
// $COPYRIGHT$
//
// Additional copyrights may follow
//
// $HEADER$
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are
// met:
//
// - Redistributions of source code must retain the above copyright
//   notice, this list of conditions and the following disclaimer.
//
// - Redistributions in binary form must reproduce the above copyright
//   notice, this list of conditions and the following disclaimer listed
//   in this license in the documentation and/or other materials
//   provided with the distribution.
//
// - Neither the name of the copyright holders nor the names of its
//   contributors may be used to endorse or promote products derived from
//   this software without specific prior written permission.
//
// The copyright holders provide no reassurances that the source code
// provided does not infringe any patent, copyright, or any other
// intellectual property rights of third parties.  The copyright holders
// disclaim any liability to any recipient for claims brought against
// recipient by any third party for infringement of that parties
// intellectual property rights.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
// "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
// LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
// A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
// OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
// SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
// LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
// DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
// THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#pragma once
#include <float.h>
#include <math.h>
#ifndef __CUDACC__
#include <stdexcept>
#endif

#ifdef __CUDACC__
#define CLIPP2_BLAS_FN __device__
#else
#define CLIPP2_BLAS_FN
#endif

namespace clipp2::resident_slsqp {
struct DeviceBlas {
    CLIPP2_BLAS_FN static void unsupported() {
#ifdef __CUDA_ARCH__
        asm volatile("trap;");
#elif !defined(__CUDACC__)
        throw std::runtime_error("Resident SLSQP linear-algebra contract violated");
#endif
    }
    CLIPP2_BLAS_FN static bool flag(const char* value, char expected) {
        return *value == expected || *value == expected + ('a'-'A');
    }
    CLIPP2_BLAS_FN static int start(int n, int stride) {
        return stride < 0 ? (1-n)*stride : 0;
    }
    CLIPP2_BLAS_FN static void daxpy(int* n, double* alpha, double* x,
                                    int* ix, double* y, int* iy) {
        if (*n <= 0 || *alpha == 0.) return;
        for (int k=0, i=start(*n,*ix), j=start(*n,*iy); k<*n; ++k,i+=*ix,j+=*iy)
            y[j] += *alpha*x[i];
    }
    CLIPP2_BLAS_FN static void dscal(int* n, double* alpha, double* x, int* ix) {
        if (*n <= 0 || *ix <= 0) return;
        for (int k=0; k<*n; ++k) x[k**ix] *= *alpha;
    }
    CLIPP2_BLAS_FN static double ddot(int* n, double* x, int* ix, double* y, int* iy) {
        double result=0.;
        for (int k=0,i=start(*n,*ix),j=start(*n,*iy); k<*n; ++k,i+=*ix,j+=*iy)
            result += x[i]*y[j];
        return result;
    }
    CLIPP2_BLAS_FN static double dnrm2(int* n, double* x, int* ix) {
        if (*n < 1 || *ix < 1) return 0.;
        double scale=0., squares=1.;
        for (int k=0; k<*n; ++k) {
            double value=fabs(x[k**ix]);
            if (isnan(value)) return value;
            if (isinf(value)) return value;
            if (value == 0.) continue;
            if (scale < value) {
                double ratio=scale/value;
                squares=1.+squares*(ratio*ratio);
                scale=value;
            } else {
                double ratio=value/scale;
                squares+=ratio*ratio;
            }
        }
        return scale*sqrt(squares);
    }
    CLIPP2_BLAS_FN static void dgemv(const char* trans, int* m, int* n,
        double* alpha, double* a, int* lda, double* x, int* ix,
        double* beta, double* y, int* iy) {
        bool normal=flag(trans,'N');
        if (!normal && !flag(trans,'T')) unsupported();
        if (*m < 0 || *n < 0 || *ix == 0 || *iy == 0 || *lda < (*m>1?*m:1)) unsupported();
        if (*m == 0 || *n == 0 || (*alpha == 0. && *beta == 1.)) return;
        int rows=normal?*m:*n, cols=normal?*n:*m;
        int sy=start(rows,*iy), sx=start(cols,*ix);
        if (*beta != 1.) for (int i=0; i<rows; ++i)
            y[sy+i**iy]=*beta==0.?0.:*beta*y[sy+i**iy];
        if (*alpha == 0.) return;
        if (normal) {
            for (int j=0; j<*n; ++j) {
                double value=*alpha*x[sx+j**ix];
                for (int i=0; i<*m; ++i) y[sy+i**iy]+=value*a[i+j**lda];
            }
        } else {
            for (int j=0; j<*n; ++j) {
                double value=0.;
                for (int i=0; i<*m; ++i) value+=a[i+j**lda]*x[sx+i**ix];
                y[sy+j**iy]+=*alpha*value;
            }
        }
    }
    CLIPP2_BLAS_FN static double at(double* a, int n, int lda, int i, int j, bool upper, bool packed) {
        if (!packed) return a[i+j*lda];
        return upper?a[j*(j+1)/2+i]:a[j*n-j*(j-1)/2+i-j];
    }
    CLIPP2_BLAS_FN static void triangular(const char* uplo,const char* trans,const char* diag,
        int n,double* a,int lda,double* x,int stride,bool packed,bool solve) {
        bool upper=flag(uplo,'U'), normal=flag(trans,'N'), unit=flag(diag,'U');
        if ((!upper&&!flag(uplo,'L')) || (!normal&&!flag(trans,'T')) ||
            (!unit&&!flag(diag,'N')) || n<0 || stride==0) unsupported();
        int sx=start(n,stride);
        if (solve) {
            bool forward=normal?!upper:upper;
            for (int k=0; k<n; ++k) {
                int j=forward?k:n-1-k;
                double value=x[sx+j*stride];
                if (normal) {
                    if (value==0.) continue;
                    if (!unit) value/=at(a,n,lda,j,j,upper,packed);
                    x[sx+j*stride]=value;
                    for (int i=j+(upper?-1:1); i>=0&&i<n; i+=(upper?-1:1))
                        x[sx+i*stride]-=value*at(a,n,lda,i,j,upper,packed);
                } else {
                    for (int i=upper?0:n-1; upper?i<j:i>j; i+=upper?1:-1)
                        value-=at(a,n,lda,i,j,upper,packed)*x[sx+i*stride];
                    if (!unit) value/=at(a,n,lda,j,j,upper,packed);
                    x[sx+j*stride]=value;
                }
            }
        } else {
            bool forward=normal?upper:!upper;
            for (int k=0; k<n; ++k) {
                int j=forward?k:n-1-k;
                double value=x[sx+j*stride];
                if (normal) {
                    if (value==0.) continue;
                    for (int i=upper?0:n-1; upper?i<j:i>j; i+=upper?1:-1)
                        x[sx+i*stride]+=value*at(a,n,lda,i,j,upper,packed);
                    if (!unit) x[sx+j*stride]*=at(a,n,lda,j,j,upper,packed);
                } else {
                    if (!unit) value*=at(a,n,lda,j,j,upper,packed);
                    for (int i=j+(upper?-1:1); i>=0&&i<n; i+=(upper?-1:1))
                        value+=at(a,n,lda,i,j,upper,packed)*x[sx+i*stride];
                    x[sx+j*stride]=value;
                }
            }
        }
    }
    CLIPP2_BLAS_FN static void dtpsv(const char*u,const char*t,const char*d,int*n,double*a,double*x,int*s) {
        triangular(u,t,d,*n,a,0,x,*s,true,true);
    }
    CLIPP2_BLAS_FN static void dtpmv(const char*u,const char*t,const char*d,int*n,double*a,double*x,int*s) {
        triangular(u,t,d,*n,a,0,x,*s,true,false);
    }
    CLIPP2_BLAS_FN static void dtrsv(const char*u,const char*t,const char*d,int*n,double*a,int*l,double*x,int*s) {
        triangular(u,t,d,*n,a,*l,x,*s,false,true);
    }
    CLIPP2_BLAS_FN static void dtrsm(const char*side,const char*u,const char*t,const char*d,
        int*m,int*n,double*alpha,double*a,int*lda,double*b,int*ldb) {
        // The preserved core calls only this variant; do not silently use a
        // different arithmetic schedule for an unsupported generalized request.
        if (!flag(side,'R')||!flag(u,'U')||!flag(t,'N')||!flag(d,'N')) unsupported();
        for (int j=0; j<*n; ++j) {
            if (*alpha != 1.) for (int i=0; i<*m; ++i) b[i+j**ldb]*=*alpha;
            for (int k=0; k<j; ++k) if (a[k+j**lda] != 0.)
                for (int i=0; i<*m; ++i) b[i+j**ldb]-=a[k+j**lda]*b[i+k**ldb];
            double inverse=1./a[j+j**lda];
            for (int i=0; i<*m; ++i) b[i+j**ldb]*=inverse;
        }
    }
    CLIPP2_BLAS_FN static double dlapy2(double x,double y) {
        if (isnan(x)) return x;
        if (isnan(y)) return y;
        x=fabs(x); y=fabs(y);
        double high=x>y?x:y, low=x>y?y:x;
        if (low==0. || isinf(high)) return high;
        double ratio=low/high;
        return high*sqrt(1.+ratio*ratio);
    }
    CLIPP2_BLAS_FN static void dlarfg(int n,double* alpha,double*x,int stride,double*tau) {
        if (n<=1) { *tau=0.; return; }
        int count=n-1;
        double norm=dnrm2(&count,x,&stride);
        if (norm==0.) { *tau=0.; return; }
        double beta=-copysign(dlapy2(*alpha,norm),*alpha);
        constexpr double safe=DBL_MIN/(DBL_EPSILON/2.);
        int scaled=0;
        if (fabs(beta)<safe) {
            double inverse=1./safe;
            do {
                ++scaled; dscal(&count,&inverse,x,&stride);
                beta*=inverse; *alpha*=inverse;
            } while (fabs(beta)<safe && scaled<20);
            norm=dnrm2(&count,x,&stride);
            beta=-copysign(dlapy2(*alpha,norm),*alpha);
        }
        *tau=(beta-*alpha)/beta;
        double inverse=1./(*alpha-beta);
        dscal(&count,&inverse,x,&stride);
        for (int k=0;k<scaled;++k) beta*=safe;
        *alpha=beta;
    }
    CLIPP2_BLAS_FN static void dlarfgp(int*n,double*alpha,double*x,int*stride,double*tau) {
        if (*n<=0) { *tau=0.; return; }
        int count=*n-1;
        double norm=dnrm2(&count,x,stride);
        if (norm<=DBL_EPSILON*fabs(*alpha)) {
            *tau=*alpha>=0.?0.:2.;
            if (*alpha<0.) { for(int i=0;i<count;++i)x[i**stride]=0.; *alpha=-*alpha; }
            return;
        }
        double beta=copysign(dlapy2(*alpha,norm),*alpha);
        constexpr double safe=DBL_MIN/(DBL_EPSILON/2.);
        int scaled=0;
        if (fabs(beta)<safe) {
            double inverse=1./safe;
            do { ++scaled; dscal(&count,&inverse,x,stride); beta*=inverse; *alpha*=inverse; }
            while (fabs(beta)<safe && scaled<20);
            norm=dnrm2(&count,x,stride); beta=copysign(dlapy2(*alpha,norm),*alpha);
        }
        double saved=*alpha;
        *alpha+=beta;
        if (beta<0.) { beta=-beta; *tau=-*alpha/beta; }
        else { *alpha=norm*(norm/ *alpha); *tau=*alpha/beta; *alpha=-*alpha; }
        if (fabs(*tau)<=safe) {
            if (saved>=0.) *tau=0.;
            else { *tau=2.; for(int i=0;i<count;++i)x[i**stride]=0.; beta=-saved; }
        } else { double inverse=1./ *alpha; dscal(&count,&inverse,x,stride); }
        for(int k=0;k<scaled;++k) beta*=safe;
        *alpha=beta;
    }
    CLIPP2_BLAS_FN static void dlartgp(double*f,double*g,double*cs,double*sn,double*r) {
        if (*g==0.) { *cs=copysign(1.,*f); *sn=0.; *r=fabs(*f); return; }
        if (*f==0.) { *cs=0.; *sn=copysign(1.,*g); *r=fabs(*g); return; }
        // DLAMCH(S)=2^-1022, DLAMCH(E)=2^-53; Fortran INT truncates
        // log2(S/E)/2=-484.5 toward zero, yielding exponent -484.
        constexpr double low=0x1p-484, high=0x1p484;
        double ff=*f,gg=*g,scale=fmax(fabs(ff),fabs(gg));
        int count=0, direction=0;
        if (scale>=high) {
            direction=1;
            do { ++count; ff*=low; gg*=low; scale=fmax(fabs(ff),fabs(gg)); }
            while (scale>=high && count<20);
        } else if (scale<=low) {
            direction=-1;
            do { ++count; ff*=high; gg*=high; scale=fmax(fabs(ff),fabs(gg)); }
            while (scale<=low);
        }
        *r=sqrt(ff*ff+gg*gg); *cs=ff/ *r; *sn=gg/ *r;
        for(int i=0;i<count;++i)*r*=direction>0?high:low;
        if (*r<0.) { *cs=-*cs; *sn=-*sn; *r=-*r; }
    }
    CLIPP2_BLAS_FN static void dlarf(const char*side,int*m,int*n,double*v,int*stride,
        double*tau,double*c,int*ldc,double*work) {
        bool left=flag(side,'L');
        if (!left&&!flag(side,'R')) unsupported();
        if (*tau==0. || *m==0 || *n==0) return;
        int lastv=left?*m:*n, origin=start(lastv,*stride);
        while(lastv>0 && v[origin+(lastv-1)**stride]==0.) --lastv;
        if (lastv==0) return;
        int lastc=left?*n:*m;
        while (lastc>0) {
            bool nonzero=false;
            for(int i=0;i<lastv;++i)
                nonzero=nonzero||(left?c[i+(lastc-1)**ldc]:c[lastc-1+i**ldc])!=0.;
            if(nonzero)break;
            --lastc;
        }
        int one=1;
        double alpha=1.,zero=0.;
        if(left)dgemv("T",&lastv,&lastc,&alpha,c,ldc,v,stride,&zero,work,&one);
        else dgemv("N",&lastc,&lastv,&alpha,c,ldc,v,stride,&zero,work,&one);
        int rows=left?lastv:lastc,cols=left?lastc:lastv;
        for(int j=0;j<cols;++j) {
            double yy=left?work[j]:v[origin+j**stride];
            if(yy==0.)continue;
            double factor=-*tau*yy;
            for(int i=0;i<rows;++i)c[i+j**ldc]+=(left?v[origin+i**stride]:work[i])*factor;
        }
    }
    CLIPP2_BLAS_FN static void dgeqr2(int*m,int*n,double*a,int*lda,double*tau,double*work,int*info) {
        *info=*m<0?-1:(*n<0?-2:(*lda<(*m>1?*m:1)?-4:0));
        if(*info!=0)unsupported();
        int k=*m<*n?*m:*n,one=1;
        for(int i=0;i<k;++i) {
            dlarfg(*m-i,&a[i+i**lda],&a[(i+1<*m?i+1:i)+i**lda],one,&tau[i]);
            if(i+1>=*n)continue;
            double saved=a[i+i**lda]; a[i+i**lda]=1.;
            int rows=*m-i,cols=*n-i-1;
            dlarf("L",&rows,&cols,&a[i+i**lda],&one,&tau[i],&a[i+(i+1)**lda],lda,work);
            a[i+i**lda]=saved;
        }
    }
    CLIPP2_BLAS_FN static void dgerq2(int*m,int*n,double*a,int*lda,double*tau,double*work,int*info) {
        *info=*m<0?-1:(*n<0?-2:(*lda<(*m>1?*m:1)?-4:0));
        if(*info!=0)unsupported();
        int k=*m<*n?*m:*n;
        for(int i=k-1;i>=0;--i) {
            int row=*m-k+i,col=*n-k+i,cols=col+1;
            dlarfg(cols,&a[row+col**lda],&a[row],*lda,&tau[i]);
            double saved=a[row+col**lda]; a[row+col**lda]=1.;
            dlarf("R",&row,&cols,&a[row],lda,&tau[i],a,lda,work);
            a[row+col**lda]=saved;
        }
    }
    CLIPP2_BLAS_FN static void apply_q(bool rq,const char*side,const char*trans,int*m,int*n,int*k,
        double*a,int*lda,double*tau,double*c,int*ldc,double*work,int*info) {
        bool left=flag(side,'L'),normal=flag(trans,'N');
        int nq=left?*m:*n;
        *info=(!left&&!flag(side,'R'))?-1:(!normal&&!flag(trans,'T'))?-2:
            *m<0?-3:*n<0?-4:(*k<0||*k>nq)?-5:
            *lda<((rq?*k:nq)>1?(rq?*k:nq):1)?-7:*ldc<(*m>1?*m:1)?-10:0;
        if(*info!=0)unsupported();
        if(*m==0||*n==0||*k==0)return;
        bool forward=(left&&!normal)||(!left&&normal);
        int one=1;
        for(int j=0;j<*k;++j) {
            int i=forward?j:*k-1-j,rows=*m,cols=*n;
            if(rq) {
                if(left)rows=*m-*k+i+1;else cols=*n-*k+i+1;
                int diagonal=i+(nq-*k+i)**lda;
                double saved=a[diagonal];a[diagonal]=1.;
                dlarf(side,&rows,&cols,&a[i],lda,&tau[i],c,ldc,work);
                a[diagonal]=saved;
            } else {
                if(left)rows-=i;else cols-=i;
                int diagonal=i+i**lda,offset=left?i:i**ldc;
                double saved=a[diagonal];a[diagonal]=1.;
                dlarf(side,&rows,&cols,&a[diagonal],&one,&tau[i],&c[offset],ldc,work);
                a[diagonal]=saved;
            }
        }
    }
    CLIPP2_BLAS_FN static void dorm2r(const char*s,const char*t,int*m,int*n,int*k,double*a,
        int*l,double*tau,double*c,int*lc,double*w,int*info) {
        apply_q(false,s,t,m,n,k,a,l,tau,c,lc,w,info);
    }
    CLIPP2_BLAS_FN static void dormr2(const char*s,const char*t,int*m,int*n,int*k,double*a,
        int*l,double*tau,double*c,int*lc,double*w,int*info) {
        apply_q(true,s,t,m,n,k,a,l,tau,c,lc,w,info);
    }
    CLIPP2_BLAS_FN static void dgelsy(int*,int*,int*,double*,int*,double*,int*,int*,
        double*,int*,double*,int*,int*) {
        // Finite production simplex bounds always add positive inequality
        // count. The complete core retains this general branch, but this
        // guarded policy cannot execute arbitrary unbounded least squares.
        unsupported();
    }
};
} // namespace clipp2::resident_slsqp
#undef CLIPP2_BLAS_FN
