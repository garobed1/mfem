// Copyright (c) 2010, Lawrence Livermore National Security, LLC. Produced at
// the Lawrence Livermore National Laboratory. LLNL-CODE-443211. All Rights
// reserved. See file COPYRIGHT for details.
//
// This file is part of the MFEM library. For more information and source code
// availability see http://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the GNU Lesser General Public License (as published by the Free
// Software Foundation) version 2.1 dated February 1999.

#include "../general/forall.hpp"
#include "bilininteg.hpp"
#include "gridfunc.hpp"

using namespace std;

namespace mfem
{

// PA Vector Diffusion Integrator

// PA Diffusion Assemble 2D kernel
static void PAVectorDiffusionSetup2D(const int Q1D,
                                     const int NE,
                                     const Array<double> &w,
                                     const Vector &j,
                                     const double COEFF,
                                     Vector &op)
{
   const int NQ = Q1D*Q1D;
   auto W = w.Read();

   auto J = Reshape(j.Read(), NQ, 2, 2, NE);
   auto y = Reshape(op.Write(), NQ, 3, NE);

   MFEM_FORALL(e, NE,
   {
      for (int q = 0; q < NQ; ++q)
      {
         const double J11 = J(q,0,0,e);
         const double J21 = J(q,1,0,e);
         const double J12 = J(q,0,1,e);
         const double J22 = J(q,1,1,e);
         const double c_detJ = W[q] * COEFF / ((J11*J22)-(J21*J12));
         y(q,0,e) =  c_detJ * (J12*J12 + J22*J22); // 1,1
         y(q,1,e) = -c_detJ * (J12*J11 + J22*J21); // 1,2
         y(q,2,e) =  c_detJ * (J11*J11 + J21*J21); // 2,2
      }
   });
}

// PA Diffusion Assemble 3D kernel
static void PAVectorDiffusionSetup3D(const int Q1D,
                                     const int NE,
                                     const Array<double> &w,
                                     const Vector &j,
                                     const double COEFF,
                                     Vector &op)
{
   const int NQ = Q1D*Q1D*Q1D;
   auto W = w.Read();
   auto J = Reshape(j.Read(), NQ, 3, 3, NE);
   auto y = Reshape(op.Write(), NQ, 6, NE);
   MFEM_FORALL(e, NE,
   {
      for (int q = 0; q < NQ; ++q)
      {
         const double J11 = J(q,0,0,e);
         const double J21 = J(q,1,0,e);
         const double J31 = J(q,2,0,e);
         const double J12 = J(q,0,1,e);
         const double J22 = J(q,1,1,e);
         const double J32 = J(q,2,1,e);
         const double J13 = J(q,0,2,e);
         const double J23 = J(q,1,2,e);
         const double J33 = J(q,2,2,e);
         const double detJ = J11 * (J22 * J33 - J32 * J23) -
         /* */               J21 * (J12 * J33 - J32 * J13) +
         /* */               J31 * (J12 * J23 - J22 * J13);
         const double c_detJ = W[q] * COEFF / detJ;
         // adj(J)
         const double A11 = (J22 * J33) - (J23 * J32);
         const double A12 = (J32 * J13) - (J12 * J33);
         const double A13 = (J12 * J23) - (J22 * J13);
         const double A21 = (J31 * J23) - (J21 * J33);
         const double A22 = (J11 * J33) - (J13 * J31);
         const double A23 = (J21 * J13) - (J11 * J23);
         const double A31 = (J21 * J32) - (J31 * J22);
         const double A32 = (J31 * J12) - (J11 * J32);
         const double A33 = (J11 * J22) - (J12 * J21);
         // detJ J^{-1} J^{-T} = (1/detJ) adj(J) adj(J)^T
         y(q,0,e) = c_detJ * (A11*A11 + A12*A12 + A13*A13); // 1,1
         y(q,1,e) = c_detJ * (A11*A21 + A12*A22 + A13*A23); // 2,1
         y(q,2,e) = c_detJ * (A11*A31 + A12*A32 + A13*A33); // 3,1
         y(q,3,e) = c_detJ * (A21*A21 + A22*A22 + A23*A23); // 2,2
         y(q,4,e) = c_detJ * (A21*A31 + A22*A32 + A23*A33); // 3,2
         y(q,5,e) = c_detJ * (A31*A31 + A32*A32 + A33*A33); // 3,3
      }
   });
}

static void PAVectorDiffusionSetup(const int dim,
                                   const int D1D,
                                   const int Q1D,
                                   const int NE,
                                   const Array<double> &W,
                                   const Vector &J,
                                   const double COEFF,
                                   Vector &op)
{
   if (dim == 1) { MFEM_ABORT("dim==1 not supported in PADiffusionSetup"); }
   if (dim == 2)
   {
      PAVectorDiffusionSetup2D(Q1D, NE, W, J, COEFF, op);
   }
   if (dim == 3)
   {
      PAVectorDiffusionSetup3D(Q1D, NE, W, J, COEFF, op);
   }
}

void VectorDiffusionIntegrator::Setup(const FiniteElementSpace &fes)
{
   // Assumes tensor-product elements
   Mesh *mesh = fes.GetMesh();
   const FiniteElement &el = *fes.GetFE(0);
   const IntegrationRule *ir
      = IntRule ? IntRule : &DiffusionIntegrator::GetRule(el, el);
   const int dims = el.GetDim();
   const int symmDims = (dims * (dims + 1)) / 2; // 1x1: 1, 2x2: 3, 3x3: 6
   const int nq = ir->GetNPoints();
   dim = mesh->Dimension();
   ne = fes.GetNE();
   geom = mesh->GetGeometricFactors(*ir, GeometricFactors::JACOBIANS);
   maps = &el.GetDofToQuad(*ir, DofToQuad::TENSOR);
   dofs1D = maps->ndof;
   quad1D = maps->nqpt;
   pa_data.SetSize(symmDims * nq * ne, Device::GetMemoryType());
   double coeff = 1.0;
   if (Q)
   {
      ConstantCoefficient *cQ = dynamic_cast<ConstantCoefficient*>(Q);
      MFEM_VERIFY(cQ != NULL, "only ConstantCoefficient is supported!");
      coeff = cQ->constant;
   }
   PAVectorDiffusionSetup(dim, dofs1D, quad1D, ne, ir->GetWeights(), geom->J,
                          coeff, pa_data);
}

// PA Diffusion Apply 2D kernel
MFEM_JIT
template<int T_D1D = 0, int T_Q1D = 0> static
void PAVectorDiffusionApply2D(const int NE,
                              const Array<double> &b,
                              const Array<double> &g,
                              const Array<double> &bt,
                              const Array<double> &gt,
                              const Vector &_op,
                              const Vector &_x,
                              Vector &_y,
                              const int d1d = 0,
                              const int q1d = 0)
{
   const int D1D = T_D1D ? T_D1D : d1d;
   const int Q1D = T_Q1D ? T_Q1D : q1d;
   constexpr int VDIM = 2;
   MFEM_VERIFY(D1D <= MAX_D1D, "");
   MFEM_VERIFY(Q1D <= MAX_Q1D, "");
   auto B = Reshape(b.Read(), Q1D, D1D);
   auto G = Reshape(g.Read(), Q1D, D1D);
   auto Bt = Reshape(bt.Read(), D1D, Q1D);
   auto Gt = Reshape(gt.Read(), D1D, Q1D);
   auto op = Reshape(_op.Read(), Q1D*Q1D, 3, NE);
   auto x = Reshape(_x.Read(), D1D, D1D, VDIM, NE);
   auto y = Reshape(_y.ReadWrite(), D1D, D1D, VDIM, NE);
   MFEM_FORALL(e, NE,
   {
      const int D1D = T_D1D ? T_D1D : d1d;
      const int Q1D = T_Q1D ? T_Q1D : q1d;
      // the following variables are evaluated at compile time
      constexpr int max_D1D = T_D1D ? T_D1D : MAX_D1D;
      constexpr int max_Q1D = T_Q1D ? T_Q1D : MAX_Q1D;

      for (int c = 0; c < VDIM; ++ c)
      {
         double grad[max_Q1D][max_Q1D][2];
         for (int qy = 0; qy < Q1D; ++qy)
         {
            for (int qx = 0; qx < Q1D; ++qx)
            {
               grad[qy][qx][0] = 0.0;
               grad[qy][qx][1] = 0.0;
            }
         }
         for (int dy = 0; dy < D1D; ++dy)
         {
            double gradX[max_Q1D][2];
            for (int qx = 0; qx < Q1D; ++qx)
            {
               gradX[qx][0] = 0.0;
               gradX[qx][1] = 0.0;
            }
            for (int dx = 0; dx < D1D; ++dx)
            {
               const double s = x(dx,dy,c,e);
               for (int qx = 0; qx < Q1D; ++qx)
               {
                  gradX[qx][0] += s * B(qx,dx);
                  gradX[qx][1] += s * G(qx,dx);
               }
            }
            for (int qy = 0; qy < Q1D; ++qy)
            {
               const double wy  = B(qy,dy);
               const double wDy = G(qy,dy);
               for (int qx = 0; qx < Q1D; ++qx)
               {
                  grad[qy][qx][0] += gradX[qx][1] * wy;
                  grad[qy][qx][1] += gradX[qx][0] * wDy;
               }
            }
         }
         // Calculate Dxy, xDy in plane
         for (int qy = 0; qy < Q1D; ++qy)
         {
            for (int qx = 0; qx < Q1D; ++qx)
            {
               const int q = qx + qy * Q1D;

               const double O11 = op(q,0,e);
               const double O12 = op(q,1,e);
               const double O22 = op(q,2,e);

               const double gradX = grad[qy][qx][0];
               const double gradY = grad[qy][qx][1];

               grad[qy][qx][0] = (O11 * gradX) + (O12 * gradY);
               grad[qy][qx][1] = (O12 * gradX) + (O22 * gradY);
            }
         }
         for (int qy = 0; qy < Q1D; ++qy)
         {
            double gradX[max_D1D][2];
            for (int dx = 0; dx < D1D; ++dx)
            {
               gradX[dx][0] = 0;
               gradX[dx][1] = 0;
            }
            for (int qx = 0; qx < Q1D; ++qx)
            {
               const double gX = grad[qy][qx][0];
               const double gY = grad[qy][qx][1];
               for (int dx = 0; dx < D1D; ++dx)
               {
                  const double wx  = Bt(dx,qx);
                  const double wDx = Gt(dx,qx);
                  gradX[dx][0] += gX * wDx;
                  gradX[dx][1] += gY * wx;
               }
            }
            for (int dy = 0; dy < D1D; ++dy)
            {
               const double wy  = Bt(dy,qy);
               const double wDy = Gt(dy,qy);
               for (int dx = 0; dx < D1D; ++dx)
               {
                  y(dx,dy,c,e) += ((gradX[dx][0] * wy) + (gradX[dx][1] * wDy));
               }
            }
         }
      }
   });
}

// PA Diffusion Apply 3D kernel
MFEM_JIT
template<const int T_D1D = 0,
         const int T_Q1D = 0> static
void PAVectorDiffusionApply3D(int NE,
                              const Array<double> &b,
                              const Array<double> &g,
                              const Array<double> &bt,
                              const Array<double> &gt,
                              const Vector &_op,
                              const Vector &_x,
                              Vector &_y,
                              const int d1d = 0,
                              const int q1d = 0)
{
   const int D1D = T_D1D ? T_D1D : d1d;
   const int Q1D = T_Q1D ? T_Q1D : q1d;
   constexpr int VDIM = 3;
   MFEM_VERIFY(D1D <= MAX_D1D, "");
   MFEM_VERIFY(Q1D <= MAX_Q1D, "");
   auto B = Reshape(b.Read(), Q1D, D1D);
   auto G = Reshape(g.Read(), Q1D, D1D);
   auto Bt = Reshape(bt.Read(), D1D, Q1D);
   auto Gt = Reshape(gt.Read(), D1D, Q1D);
   auto op = Reshape(_op.Read(), Q1D*Q1D*Q1D, 6, NE);
   auto x = Reshape(_x.Read(), D1D, D1D, D1D, VDIM, NE);
   auto y = Reshape(_y.ReadWrite(), D1D, D1D, D1D, VDIM, NE);
   MFEM_FORALL(e, NE,
   {
      const int D1D = T_D1D ? T_D1D : d1d;
      const int Q1D = T_Q1D ? T_Q1D : q1d;
      constexpr int max_D1D = T_D1D ? T_D1D : MAX_D1D;
      constexpr int max_Q1D = T_Q1D ? T_Q1D : MAX_Q1D;
      for (int c = 0; c < VDIM; ++ c)
      {
         double grad[max_Q1D][max_Q1D][max_Q1D][3];
         for (int qz = 0; qz < Q1D; ++qz)
         {
            for (int qy = 0; qy < Q1D; ++qy)
            {
               for (int qx = 0; qx < Q1D; ++qx)
               {
                  grad[qz][qy][qx][0] = 0.0;
                  grad[qz][qy][qx][1] = 0.0;
                  grad[qz][qy][qx][2] = 0.0;
               }
            }
         }
         for (int dz = 0; dz < D1D; ++dz)
         {
            double gradXY[max_Q1D][max_Q1D][3];
            for (int qy = 0; qy < Q1D; ++qy)
            {
               for (int qx = 0; qx < Q1D; ++qx)
               {
                  gradXY[qy][qx][0] = 0.0;
                  gradXY[qy][qx][1] = 0.0;
                  gradXY[qy][qx][2] = 0.0;
               }
            }
            for (int dy = 0; dy < D1D; ++dy)
            {
               double gradX[max_Q1D][2];
               for (int qx = 0; qx < Q1D; ++qx)
               {
                  gradX[qx][0] = 0.0;
                  gradX[qx][1] = 0.0;
               }
               for (int dx = 0; dx < D1D; ++dx)
               {
                  const double s = x(dx,dy,dz,c,e);
                  for (int qx = 0; qx < Q1D; ++qx)
                  {
                     gradX[qx][0] += s * B(qx,dx);
                     gradX[qx][1] += s * G(qx,dx);
                  }
               }
               for (int qy = 0; qy < Q1D; ++qy)
               {
                  const double wy  = B(qy,dy);
                  const double wDy = G(qy,dy);
                  for (int qx = 0; qx < Q1D; ++qx)
                  {
                     const double wx  = gradX[qx][0];
                     const double wDx = gradX[qx][1];
                     gradXY[qy][qx][0] += wDx * wy;
                     gradXY[qy][qx][1] += wx  * wDy;
                     gradXY[qy][qx][2] += wx  * wy;
                  }
               }
            }
            for (int qz = 0; qz < Q1D; ++qz)
            {
               const double wz  = B(qz,dz);
               const double wDz = G(qz,dz);
               for (int qy = 0; qy < Q1D; ++qy)
               {
                  for (int qx = 0; qx < Q1D; ++qx)
                  {
                     grad[qz][qy][qx][0] += gradXY[qy][qx][0] * wz;
                     grad[qz][qy][qx][1] += gradXY[qy][qx][1] * wz;
                     grad[qz][qy][qx][2] += gradXY[qy][qx][2] * wDz;
                  }
               }
            }
         }
         // Calculate Dxyz, xDyz, xyDz in plane
         for (int qz = 0; qz < Q1D; ++qz)
         {
            for (int qy = 0; qy < Q1D; ++qy)
            {
               for (int qx = 0; qx < Q1D; ++qx)
               {
                  const int q = qx + (qy + qz * Q1D) * Q1D;
                  const double O11 = op(q,0,e);
                  const double O12 = op(q,1,e);
                  const double O13 = op(q,2,e);
                  const double O22 = op(q,3,e);
                  const double O23 = op(q,4,e);
                  const double O33 = op(q,5,e);
                  const double gradX = grad[qz][qy][qx][0];
                  const double gradY = grad[qz][qy][qx][1];
                  const double gradZ = grad[qz][qy][qx][2];
                  grad[qz][qy][qx][0] = (O11*gradX)+(O12*gradY)+(O13*gradZ);
                  grad[qz][qy][qx][1] = (O12*gradX)+(O22*gradY)+(O23*gradZ);
                  grad[qz][qy][qx][2] = (O13*gradX)+(O23*gradY)+(O33*gradZ);
               }
            }
         }
         for (int qz = 0; qz < Q1D; ++qz)
         {
            double gradXY[max_D1D][max_D1D][3];
            for (int dy = 0; dy < D1D; ++dy)
            {
               for (int dx = 0; dx < D1D; ++dx)
               {
                  gradXY[dy][dx][0] = 0;
                  gradXY[dy][dx][1] = 0;
                  gradXY[dy][dx][2] = 0;
               }
            }
            for (int qy = 0; qy < Q1D; ++qy)
            {
               double gradX[max_D1D][3];
               for (int dx = 0; dx < D1D; ++dx)
               {
                  gradX[dx][0] = 0;
                  gradX[dx][1] = 0;
                  gradX[dx][2] = 0;
               }
               for (int qx = 0; qx < Q1D; ++qx)
               {
                  const double gX = grad[qz][qy][qx][0];
                  const double gY = grad[qz][qy][qx][1];
                  const double gZ = grad[qz][qy][qx][2];
                  for (int dx = 0; dx < D1D; ++dx)
                  {
                     const double wx  = Bt(dx,qx);
                     const double wDx = Gt(dx,qx);
                     gradX[dx][0] += gX * wDx;
                     gradX[dx][1] += gY * wx;
                     gradX[dx][2] += gZ * wx;
                  }
               }
               for (int dy = 0; dy < D1D; ++dy)
               {
                  const double wy  = Bt(dy,qy);
                  const double wDy = Gt(dy,qy);
                  for (int dx = 0; dx < D1D; ++dx)
                  {
                     gradXY[dy][dx][0] += gradX[dx][0] * wy;
                     gradXY[dy][dx][1] += gradX[dx][1] * wDy;
                     gradXY[dy][dx][2] += gradX[dx][2] * wy;
                  }
               }
            }
            for (int dz = 0; dz < D1D; ++dz)
            {
               const double wz  = Bt(dz,qz);
               const double wDz = Gt(dz,qz);
               for (int dy = 0; dy < D1D; ++dy)
               {
                  for (int dx = 0; dx < D1D; ++dx)
                  {
                     y(dx,dy,dz,c,e) +=
                        ((gradXY[dy][dx][0] * wz) +
                         (gradXY[dy][dx][1] * wz) +
                         (gradXY[dy][dx][2] * wDz));
                  }
               }
            }
         }
      }
   });
}

// For SmemPADiffusionApply3D, half of B and G are stored
// in shared to get B, Bt, G and Gt.
// Indices computation for SmemPADiffusionApply3D
MFEM_EMBED
inline int qi(const int q, const int d, const int Q)
{
   return (q<=d) ? q : Q-1-q;
}

MFEM_EMBED
inline int dj(const int q, const int d, const int D)
{
   return (q<=d) ? d : D-1-d;
}

MFEM_EMBED
inline int qk(const int q, const int d, const int Q)
{
   return (q<=d) ? Q-1-q : q;
}

MFEM_EMBED
inline int dl(const int q, const int d, const int D)
{
   return (q<=d) ? D-1-d : d;
}

MFEM_EMBED
inline double sign(const int q, const int d)
{
   return (q<=d) ? -1.0 : 1.0;
}

// Shared memory PA Vector Diffusion Apply 3D kernel
MFEM_JIT
template<int T_D1D = 0, int T_Q1D = 0> static
void SmemPAVectorDiffusionApply3D(const int NE,
                                  const Array<double> &b_,
                                  const Array<double> &g_,
                                  const Vector &d_,
                                  const Vector &x_,
                                  Vector &y_,
                                  const int d1d = 0,
                                  const int q1d = 0)
{
   constexpr int VDIM = 3;
   const int D1D = T_D1D ? T_D1D : d1d;
   const int Q1D = T_Q1D ? T_Q1D : q1d;
   MFEM_VERIFY(D1D+1 == Q1D, "");
   auto b = Reshape(b_.Read(), Q1D, D1D);
   auto g = Reshape(g_.Read(), Q1D, D1D);
   auto d = Reshape(d_.Read(), Q1D*Q1D*Q1D, 6, NE);
   auto x = Reshape(x_.Read(), D1D, D1D, D1D, VDIM, NE);
   auto y = Reshape(y_.ReadWrite(), D1D, D1D, D1D, VDIM, NE);
   MFEM_FORALL_3D(e, NE, Q1D, Q1D, Q1D,
   {
      const int tidz = MFEM_THREAD_ID(z);
      const int D1D = T_D1D ? T_D1D : d1d;
      const int Q1D = T_Q1D ? T_Q1D : q1d;
      constexpr int MQ1 = T_Q1D ? T_Q1D : 1;
      constexpr int MD1 = T_D1D ? T_D1D : 1;
      MFEM_SHARED double BG[MQ1*MD1];
      double (*B)[MD1] = (double (*)[MD1]) (BG);
      double (*G)[MD1] = (double (*)[MD1]) (BG);
      double (*Bt)[MQ1] = (double (*)[MQ1]) (BG);
      double (*Gt)[MQ1] = (double (*)[MQ1]) (BG);
      MFEM_SHARED double sm0[3][MQ1*MQ1*MQ1];
      MFEM_SHARED double sm1[3][MQ1*MQ1*MQ1];
      double (*X)[MD1][MD1]    = (double (*)[MD1][MD1]) (sm0+2);
      double (*DDQ0)[MD1][MQ1] = (double (*)[MD1][MQ1]) (sm0+0);
      double (*DDQ1)[MD1][MQ1] = (double (*)[MD1][MQ1]) (sm0+1);
      double (*DQQ0)[MQ1][MQ1] = (double (*)[MQ1][MQ1]) (sm1+0);
      double (*DQQ1)[MQ1][MQ1] = (double (*)[MQ1][MQ1]) (sm1+1);
      double (*DQQ2)[MQ1][MQ1] = (double (*)[MQ1][MQ1]) (sm1+2);
      double (*QQQ0)[MQ1][MQ1] = (double (*)[MQ1][MQ1]) (sm0+0);
      double (*QQQ1)[MQ1][MQ1] = (double (*)[MQ1][MQ1]) (sm0+1);
      double (*QQQ2)[MQ1][MQ1] = (double (*)[MQ1][MQ1]) (sm0+2);
      double (*QQD0)[MQ1][MD1] = (double (*)[MQ1][MD1]) (sm1+0);
      double (*QQD1)[MQ1][MD1] = (double (*)[MQ1][MD1]) (sm1+1);
      double (*QQD2)[MQ1][MD1] = (double (*)[MQ1][MD1]) (sm1+2);
      double (*QDD0)[MD1][MD1] = (double (*)[MD1][MD1]) (sm0+0);
      double (*QDD1)[MD1][MD1] = (double (*)[MD1][MD1]) (sm0+1);
      double (*QDD2)[MD1][MD1] = (double (*)[MD1][MD1]) (sm0+2);
      for (int c = 0; c < VDIM; ++ c)
      {
         MFEM_FOREACH_THREAD(dz,z,D1D)
         {
            MFEM_FOREACH_THREAD(dy,y,D1D)
            {
               MFEM_FOREACH_THREAD(dx,x,D1D)
               {
                  X[dz][dy][dx] = x(dx,dy,dz,c,e);
               }
            }
         }
         if (tidz == 0)
         {
            MFEM_FOREACH_THREAD(q,x,Q1D)
            {
               MFEM_FOREACH_THREAD(d,y,D1D)
               {
                  const int i = qi(q,d,Q1D);
                  const int j = dj(q,d,D1D);
                  const int k = qk(q,d,Q1D);
                  const int l = dl(q,d,D1D);
                  B[i][j] = b(q,d);
                  G[k][l] = g(q,d) * sign(q,d);
               }
            }
         }
         MFEM_SYNC_THREAD;
         MFEM_FOREACH_THREAD(dz,z,D1D)
         {
            MFEM_FOREACH_THREAD(dy,y,D1D)
            {
               MFEM_FOREACH_THREAD(qx,x,Q1D)
               {
                  double u = 0.0;
                  double v = 0.0;
                  for (int dx = 0; dx < D1D; ++dx)
                  {
                     const int i = qi(qx,dx,Q1D);
                     const int j = dj(qx,dx,D1D);
                     const int k = qk(qx,dx,Q1D);
                     const int l = dl(qx,dx,D1D);
                     const double coords = X[dz][dy][dx];
                     u += coords * B[i][j];
                     v += coords * G[k][l] * sign(qx,dx);
                  }
                  DDQ0[dz][dy][qx] = u;
                  DDQ1[dz][dy][qx] = v;
               }
            }
         }
         MFEM_SYNC_THREAD;
         MFEM_FOREACH_THREAD(dz,z,D1D)
         {
            MFEM_FOREACH_THREAD(qy,y,Q1D)
            {
               MFEM_FOREACH_THREAD(qx,x,Q1D)
               {
                  double u = 0.0;
                  double v = 0.0;
                  double w = 0.0;
                  for (int dy = 0; dy < D1D; ++dy)
                  {
                     const int i = qi(qy,dy,Q1D);
                     const int j = dj(qy,dy,D1D);
                     const int k = qk(qy,dy,Q1D);
                     const int l = dl(qy,dy,D1D);
                     u += DDQ1[dz][dy][qx] * B[i][j];
                     v += DDQ0[dz][dy][qx] * G[k][l] * sign(qy,dy);
                     w += DDQ0[dz][dy][qx] * B[i][j];
                  }
                  DQQ0[dz][qy][qx] = u;
                  DQQ1[dz][qy][qx] = v;
                  DQQ2[dz][qy][qx] = w;
               }
            }
         }
         MFEM_SYNC_THREAD;
         MFEM_FOREACH_THREAD(qz,z,Q1D)
         {
            MFEM_FOREACH_THREAD(qy,y,Q1D)
            {
               MFEM_FOREACH_THREAD(qx,x,Q1D)
               {
                  double u = 0.0;
                  double v = 0.0;
                  double w = 0.0;
                  for (int dz = 0; dz < D1D; ++dz)
                  {
                     const int i = qi(qz,dz,Q1D);
                     const int j = dj(qz,dz,D1D);
                     const int k = qk(qz,dz,Q1D);
                     const int l = dl(qz,dz,D1D);
                     u += DQQ0[dz][qy][qx] * B[i][j];
                     v += DQQ1[dz][qy][qx] * B[i][j];
                     w += DQQ2[dz][qy][qx] * G[k][l] * sign(qz,dz);
                  }
                  QQQ0[qz][qy][qx] = u;
                  QQQ1[qz][qy][qx] = v;
                  QQQ2[qz][qy][qx] = w;
               }
            }
         }
         MFEM_SYNC_THREAD;
         MFEM_FOREACH_THREAD(qz,z,Q1D)
         {
            MFEM_FOREACH_THREAD(qy,y,Q1D)
            {
               MFEM_FOREACH_THREAD(qx,x,Q1D)
               {
                  const int q = qx + ((qy*Q1D) + (qz*Q1D*Q1D));
                  const double O11 = d(q,0,e);
                  const double O12 = d(q,1,e);
                  const double O13 = d(q,2,e);
                  const double O22 = d(q,3,e);
                  const double O23 = d(q,4,e);
                  const double O33 = d(q,5,e);
                  const double gX = QQQ0[qz][qy][qx];
                  const double gY = QQQ1[qz][qy][qx];
                  const double gZ = QQQ2[qz][qy][qx];
                  QQQ0[qz][qy][qx] = (O11*gX) + (O12*gY) + (O13*gZ);
                  QQQ1[qz][qy][qx] = (O12*gX) + (O22*gY) + (O23*gZ);
                  QQQ2[qz][qy][qx] = (O13*gX) + (O23*gY) + (O33*gZ);
               }
            }
         }
         MFEM_SYNC_THREAD;
         if (tidz == 0)
         {
            MFEM_FOREACH_THREAD(d,y,D1D)
            {
               MFEM_FOREACH_THREAD(q,x,Q1D)
               {
                  const int i = qi(q,d,Q1D);
                  const int j = dj(q,d,D1D);
                  const int k = qk(q,d,Q1D);
                  const int l = dl(q,d,D1D);
                  Bt[j][i] = b(q,d);
                  Gt[l][k] = g(q,d) * sign(q,d);
               }
            }
         }
         MFEM_SYNC_THREAD;
         MFEM_FOREACH_THREAD(qz,z,Q1D)
         {
            MFEM_FOREACH_THREAD(qy,y,Q1D)
            {
               MFEM_FOREACH_THREAD(dx,x,D1D)
               {
                  double u = 0.0;
                  double v = 0.0;
                  double w = 0.0;
                  for (int qx = 0; qx < Q1D; ++qx)
                  {
                     const int i = qi(qx,dx,Q1D);
                     const int j = dj(qx,dx,D1D);
                     const int k = qk(qx,dx,Q1D);
                     const int l = dl(qx,dx,D1D);
                     u += QQQ0[qz][qy][qx] * Gt[l][k] * sign(qx,dx);
                     v += QQQ1[qz][qy][qx] * Bt[j][i];
                     w += QQQ2[qz][qy][qx] * Bt[j][i];
                  }
                  QQD0[qz][qy][dx] = u;
                  QQD1[qz][qy][dx] = v;
                  QQD2[qz][qy][dx] = w;
               }
            }
         }
         MFEM_SYNC_THREAD;
         MFEM_FOREACH_THREAD(qz,z,Q1D)
         {
            MFEM_FOREACH_THREAD(dy,y,D1D)
            {
               MFEM_FOREACH_THREAD(dx,x,D1D)
               {
                  double u = 0.0;
                  double v = 0.0;
                  double w = 0.0;
                  for (int qy = 0; qy < Q1D; ++qy)
                  {
                     const int i = qi(qy,dy,Q1D);
                     const int j = dj(qy,dy,D1D);
                     const int k = qk(qy,dy,Q1D);
                     const int l = dl(qy,dy,D1D);
                     u += QQD0[qz][qy][dx] * Bt[j][i];
                     v += QQD1[qz][qy][dx] * Gt[l][k] * sign(qy,dy);
                     w += QQD2[qz][qy][dx] * Bt[j][i];
                  }
                  QDD0[qz][dy][dx] = u;
                  QDD1[qz][dy][dx] = v;
                  QDD2[qz][dy][dx] = w;
               }
            }
         }
         MFEM_SYNC_THREAD;
         MFEM_FOREACH_THREAD(dz,z,D1D)
         {
            MFEM_FOREACH_THREAD(dy,y,D1D)
            {
               MFEM_FOREACH_THREAD(dx,x,D1D)
               {
                  double u = 0.0;
                  double v = 0.0;
                  double w = 0.0;
                  for (int qz = 0; qz < Q1D; ++qz)
                  {
                     const int i = qi(qz,dz,Q1D);
                     const int j = dj(qz,dz,D1D);
                     const int k = qk(qz,dz,Q1D);
                     const int l = dl(qz,dz,D1D);
                     u += QDD0[qz][dy][dx] * Bt[j][i];
                     v += QDD1[qz][dy][dx] * Bt[j][i];
                     w += QDD2[qz][dy][dx] * Gt[l][k] * sign(qz,dz);
                  }
                  y(dx,dy,dz,c,e) += (u + v + w);
               }
            }
         }
      }
      MFEM_SYNC_THREAD;
   });
}

static void PAVectorDiffusionApply(const int dim,
                                   const int D1D,
                                   const int Q1D,
                                   const int NE,
                                   const Array<double> &B,
                                   const Array<double> &G,
                                   const Array<double> &Bt,
                                   const Array<double> &Gt,
                                   const Vector &op,
                                   const Vector &x,
                                   Vector &y)
{
#ifndef MFEM_USE_JIT
   if (dim == 2)
   {
      switch ((D1D << 4 ) | Q1D)
      {
         case 0x22: return PAVectorDiffusionApply2D<2,2>(NE,B,G,Bt,Gt,op,x,y);
         case 0x33: return PAVectorDiffusionApply2D<3,3>(NE,B,G,Bt,Gt,op,x,y);
         case 0x44: return PAVectorDiffusionApply2D<4,4>(NE,B,G,Bt,Gt,op,x,y);
         case 0x55: return PAVectorDiffusionApply2D<5,5>(NE,B,G,Bt,Gt,op,x,y);
         case 0x66: return PAVectorDiffusionApply2D<6,6>(NE,B,G,Bt,Gt,op,x,y);
         case 0x77: return PAVectorDiffusionApply2D<7,7>(NE,B,G,Bt,Gt,op,x,y);
         case 0x88: return PAVectorDiffusionApply2D<8,8>(NE,B,G,Bt,Gt,op,x,y);
         case 0x99: return PAVectorDiffusionApply2D<9,9>(NE,B,G,Bt,Gt,op,x,y);
         default:   return PAVectorDiffusionApply2D(NE,B,G,Bt,Gt,op,x,y,D1D,Q1D);
      }
   }
   if (dim == 3)
   {
      switch ((D1D << 4 ) | Q1D)
      {
         case 0x23: return PAVectorDiffusionApply3D<2,3>(NE,B,G,Bt,Gt,op,x,y);
         case 0x34: return PAVectorDiffusionApply3D<3,4>(NE,B,G,Bt,Gt,op,x,y);
         case 0x45: return PAVectorDiffusionApply3D<4,5>(NE,B,G,Bt,Gt,op,x,y);
         case 0x56: return PAVectorDiffusionApply3D<5,6>(NE,B,G,Bt,Gt,op,x,y);
         case 0x67: return PAVectorDiffusionApply3D<6,7>(NE,B,G,Bt,Gt,op,x,y);
         case 0x78: return PAVectorDiffusionApply3D<7,8>(NE,B,G,Bt,Gt,op,x,y);
         case 0x89: return PAVectorDiffusionApply3D<8,9>(NE,B,G,Bt,Gt,op,x,y);
         default:   return PAVectorDiffusionApply3D(NE,B,G,Bt,Gt,op,x,y,D1D,Q1D);
      }
   }
#else // MFEM_USE_JIT
   if (dim == 2)
   {
      return PAVectorDiffusionApply2D(NE,B,G,Bt,Gt,op,x,y,D1D,Q1D);
   }
   if (dim == 3)
   {
      //return PAVectorDiffusionApply3D(NE,B,G,Bt,Gt,op,x,y,D1D,Q1D);
      return SmemPAVectorDiffusionApply3D(NE,B,G,op,x,y,D1D,Q1D);
   }
#endif // MFEM_USE_JIT
   MFEM_ABORT("Unknown kernel.");
}

// PA Diffusion Apply kernel
void VectorDiffusionIntegrator::AddMultPA(const Vector &x, Vector &y) const
{
   PAVectorDiffusionApply(dim, dofs1D, quad1D, ne,
                          maps->B, maps->G, maps->Bt, maps->Gt,
                          pa_data, x, y);
}

} // namespace mfem
