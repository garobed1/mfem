
// Testing to see convergence of mesh that conforms/does not conform to physical
// interfaces, using both h and p refinement
//
// This miniapp solves a simple 2D magnetostatic problem.
//
//                     Curl 1/mu Curl A = J + Curl mu0/mu M
//
// The permeability function is piecewise, left half of a square domain a 
// given permeability, and right half free space
//
// boundary conditions periodic from top to bottom
//
// We discretize the vector potential with H(Curl) finite elements. The magnetic
// flux B is discretized with H(Div) finite elements.
//
// Compile with: make conform
//

#include "tesla_solver.hpp"
#include <fstream>
#include <iostream>

using namespace std;
using namespace mfem;
using namespace mfem::electromagnetics;

// Permeability Function
Coefficient * SetupInvPermeabilityCoefficient();

static Vector ms_params_(0);  // Just Permeability

int h_num;

static Vector pw_mu_(0);

double magnetic_shell(const Vector &);
double magnetic_shell_inv(const Vector & x) { return 1.0/magnetic_shell(x); }

// Current Density Function
static Vector cr_params_(0);  // Axis Start, Axis End, Inner Ring Radius,
//                               Outer Ring Radius, and Total Current
//                               of current ring (annulus)
void current_ring(const Vector &, Vector &);

// Magnetization
static Vector bm_params_(0);  // Axis Start, Axis End, Bar Radius,
//                               and Magnetic Field Magnitude
void bar_magnet(const Vector &, Vector &);

static Vector ha_params_(0);  // Bounding box,
//                               axis index (0->'x', 1->'y', 2->'z'),
//                               rotation axis index
//                               and number of segments
void halbach_array(const Vector &, Vector &);

// A Field Boundary Condition for B = (Bx,By,Bz)
static Vector b_uniform_(0);
void a_bc_uniform(const Vector &, Vector&);

// Phi_M Boundary Condition for H = (0,0,1)
double phi_m_bc_uniform(const Vector &x);

// Prints the program's logo to the given output stream
void display_banner(ostream & os);

int main(int argc, char *argv[])
{
   MPI_Session mpi(argc, argv);

   if ( mpi.Root() ) { display_banner(cout); }

   // Parse command-line options.
   // const char *mesh_file = "../../data/ball-nurbs.mesh";
   int order = 1;
   int maxit = 100;
   int serial_ref_levels = 0;
   int parallel_ref_levels = 0;
   bool visualization = true;
   bool visit = true;

   Array<int> kbcs;
   Array<int> vbcs;

   Vector vbcv;

   OptionsParser args(argc, argv);
   args.AddOption(&h_num, "-hn", "--h-num",
                  "Number of elements horizontally.");
   args.AddOption(&order, "-o", "--order",
                  "Finite element order (polynomial degree).");
   args.AddOption(&pw_mu_, "-pwm", "--piecewise-mu",
                  "Piecewise values of Permeability");
   args.AddOption(&ms_params_, "-ms", "--magnetic-shell-params",
                  "Center, Inner Radius, Outer Radius, and Permeability of Magnetic Shell");
   args.AddOption(&cr_params_, "-cr", "--current-ring-params",
                  "Axis End Points, Inner Radius, Outer Radius and Total Current of Annulus");
   args.AddOption(&bm_params_, "-bm", "--bar-magnet-params",
                  "Axis End Points, Radius, and Magnetic Field of Cylindrical Magnet");
   args.AddOption(&ha_params_, "-ha", "--halbach-array-params",
                  "Bounding Box Corners and Number of Segments");
   args.AddOption(&kbcs, "-kbcs", "--surface-current-bc",
                  "Surfaces for the Surface Current (K) Boundary Condition");
   args.AddOption(&vbcs, "-vbcs", "--voltage-bc-surf",
                  "Voltage Boundary Condition Surfaces (to drive K)");
   args.AddOption(&vbcv, "-vbcv", "--voltage-bc-vals",
                  "Voltage Boundary Condition Values (to drive K)");
   // args.AddOption(&maxit, "-maxit", "--max-amr-iterations",
   //                "Max number of iterations in the main AMR loop.");
   args.AddOption(&visualization, "-vis", "--visualization", "-no-vis",
                  "--no-visualization",
                  "Enable or disable GLVis visualization.");
   args.AddOption(&visit, "-visit", "--visit", "-no-visit", "--no-visit",
                  "Enable or disable VisIt visualization.");
   args.Parse();
   if (!args.Good())
   {
      if (mpi.Root())
      {
         args.PrintUsage(cout);
      }
      return 1;
   }
   if (mpi.Root())
   {
      args.PrintOptions(cout);
   }

   // Generate structured unit square mesh
   int v = 4;
   int Nelem = h_num*(v-1)*2; //h*w*2 triangles
   int Nvert = (h_num+1)*(v);

   Mesh *mesh = new Mesh(2, Nvert, Nelem);

   double tri_v[Nvert][3];
   int tri_e[Nelem][3];

   for (int j = 0; j < h_num+1; j++)
   {
      for (int k = 0; k < v; k++)
      {
         tri_v[j*v + k][0] = (double)j / ((double)h_num);
         tri_v[j*v + k][1] = (double)k / (v-1);
         tri_v[j*v + k][2] = 0;
      }
   }

   int count = 0;
   for (int j = 0; j < h_num; j++)
   {
      for (int k = 0; k < (v-1); k++)
      {
         tri_e[j*(v-1)*2 + 2*k][0] = j*(v) + k;
         tri_e[j*(v-1)*2 + 2*k][1] = j*(v) + k + 1;
         tri_e[j*(v-1)*2 + 2*k][2] = (j+1)*(v) + k;

         tri_e[j*(v-1)*2 + 2*k+1][0] = j*(v) + k + 1;
         tri_e[j*(v-1)*2 + 2*k+1][1] = (j+1)*(v) + k;
         tri_e[j*(v-1)*2 + 2*k+1][2] = (j+1)*(v) + k + 1;
         count = count + 1;
      }
   }

   for (int j = 0; j < Nvert; j++)
   {
      mesh->AddVertex(tri_v[j]);
   }
   for (int j = 0; j < Nelem; j++)
   {
      int attribute = j + 1;
      mesh->AddTriangle(tri_e[j], attribute);
   }

   mesh->FinalizeTriMesh(1, 1, true);

   // ofstream mesh_ofs("test.mesh");
   // mesh_ofs.precision(8);
   // mesh->Print(mesh_ofs);
   {
      ofstream omesh_ofs("test_mesh.vtk");
      omesh_ofs.precision(8);
      mesh->PrintVTK(omesh_ofs, 0);
   }


   if (mpi.Root())
   {
      cout << "Starting initialization." << endl;
   }


   // Ensure that quad and hex meshes are treated as non-conforming.
   //mesh->EnsureNCMesh();

   // Refine the serial mesh on all processors to increase the resolution. In
   // this example we do 'ref_levels' of uniform refinement.
   // for (int l = 0; l < serial_ref_levels; l++)
   // {
   //    mesh->UniformRefinement();
   // }

   // Define a parallel mesh by a partitioning of the serial mesh. Refine
   // this mesh further in parallel to increase the resolution. Once the
   // parallel mesh is defined, the serial mesh can be deleted.
   ParMesh pmesh(MPI_COMM_WORLD, *mesh);
   delete mesh;

   // Refine this mesh in parallel to increase the resolution.
   // int par_ref_levels = parallel_ref_levels;
   // for (int l = 0; l < par_ref_levels; l++)
   // {
   //    pmesh.UniformRefinement();
   // }
   // Make sure tet-only meshes are marked for local refinement.
   pmesh.Finalize(true);

   // If values for Voltage BCs were not set issue a warning and exit
   if ( ( vbcs.Size() > 0 && kbcs.Size() == 0 ) ||
        ( kbcs.Size() > 0 && vbcs.Size() == 0 ) ||
        ( vbcv.Size() < vbcs.Size() ) )
   {
      if ( mpi.Root() )
      {
         cout << "The surface current (K) boundary condition requires "
              << "surface current boundary condition surfaces (with -kbcs), "
              << "voltage boundary condition surface (with -vbcs), "
              << "and voltage boundary condition values (with -vbcv)."
              << endl;
      }
      return 3;
   }

   // Create a coefficient describing the magnetic permeability
   Coefficient * muInvCoef = SetupInvPermeabilityCoefficient();

   // Create the Magnetostatic solver
   TeslaSolver Tesla(pmesh, order, kbcs, vbcs, vbcv, *muInvCoef,
                     (b_uniform_.Size() > 0 ) ? a_bc_uniform  : NULL,
                     (cr_params_.Size() > 0 ) ? current_ring  : NULL,
                     (bm_params_.Size() > 0 ) ? bar_magnet    :
                     (ha_params_.Size() > 0 ) ? halbach_array : NULL);

   // Initialize GLVis visualization
   if (visualization)
   {
      Tesla.InitializeGLVis();
   }

   // Initialize VisIt visualization
   VisItDataCollection visit_dc("Tesla-AMR-Parallel", &pmesh);

   if ( visit )
   {
      Tesla.RegisterVisItFields(visit_dc);
   }
   if (mpi.Root()) { cout << "Initialization done." << endl; }

   // The main AMR loop. In each iteration we solve the problem on the current
   // mesh, visualize the solution, estimate the error on all elements, refine
   // the worst elements and update all objects to work with the new mesh. We
   // refine until the maximum number of dofs in the Nedelec finite element
   // space reaches 10 million.
   const int max_dofs = 10000000;
   for (int it = 1; it <= maxit; it++)
   {
      if (mpi.Root())
      {
         cout << "\nAMR Iteration " << it << endl;
      }

      // Display the current number of DoFs in each finite element space
      Tesla.PrintSizes();

      // Assemble all forms
      Tesla.Assemble();

      // Solve the system and compute any auxiliary fields
      Tesla.Solve();

      // Determine the current size of the linear system
      int prob_size = Tesla.GetProblemSize();

      // Write fields to disk for VisIt
      if ( visit )
      {
         Tesla.WriteVisItFields(it);
      }

      // Send the solution by socket to a GLVis server.
      if (visualization)
      {
         Tesla.DisplayToGLVis();
      }

      if (mpi.Root())
      {
         cout << "AMR iteration " << it << " complete." << endl;
      }

      // Check stopping criteria
      if (prob_size > max_dofs)
      {
         if (mpi.Root())
         {
            cout << "Reached maximum number of dofs, exiting..." << endl;
         }
         break;
      }
      if ( it == maxit )
      {
         break;
      }

      // Wait for user input. Ask every 10th iteration.
      char c = 'c';
      if (mpi.Root() && (it % 10 == 0))
      {
         cout << "press (q)uit or (c)ontinue --> " << flush;
         cin >> c;
      }
      MPI_Bcast(&c, 1, MPI_CHAR, 0, MPI_COMM_WORLD);

      if (c != 'c')
      {
         break;
      }

      // Estimate element errors using the Zienkiewicz-Zhu error estimator.
      Vector errors(pmesh.GetNE());
      Tesla.GetErrorEstimates(errors);

      double local_max_err = errors.Max();
      double global_max_err;
      MPI_Allreduce(&local_max_err, &global_max_err, 1,
                    MPI_DOUBLE, MPI_MAX, pmesh.GetComm());

      // Refine the elements whose error is larger than a fraction of the
      // maximum element error.
      const double frac = 0.5;
      double threshold = frac * global_max_err;
      if (mpi.Root()) { cout << "Refining ..." << endl; }
      pmesh.RefineByError(errors, threshold);

      // Update the magnetostatic solver to reflect the new state of the mesh.
      Tesla.Update();

      if (pmesh.Nonconforming() && mpi.WorldSize() > 1)
      {
         if (mpi.Root()) { cout << "Rebalancing ..." << endl; }
         pmesh.Rebalance();

         // Update again after rebalancing
         Tesla.Update();
      }
   }

   delete muInvCoef;

   #endif
   return 0;
}

// Print the Volta ascii logo to the given ostream
void display_banner(ostream & os)
{
   os << "  ___________            __            " << endl
      << "  \\__    ___/___   _____|  | _____     " << endl
      << "    |    |_/ __ \\ /  ___/  | \\__  \\    " << endl
      << "    |    |\\  ___/ \\___ \\|  |__/ __ \\_  " << endl
      << "    |____| \\___  >____  >____(____  /  " << endl
      << "               \\/     \\/          \\/   " << endl << flush;
}

// The Permeability is a required coefficient which may be defined in
// various ways so we'll determine the appropriate coefficient type here.
Coefficient *
SetupInvPermeabilityCoefficient()
{
   Coefficient * coef = NULL;

   coef = new FunctionCoefficient(magnetic_shell_inv);
   
   return coef;
}

// A spherical shell with constant permeability.  The sphere has inner
// and outer radii, center, and relative permeability specified on the
// command line and stored in ms_params_.

//CHANGING FOR LEFT AND RIGHT HALF OF 2D DOMAIN
//GIVE 2 OPTIONS FOR 2D DOMAIN
double magnetic_shell(const Vector &x)
{

   if ( x(0) <= .5)
   {
      return mu0_*ms_params_(x.Size()+2);
   }
   return mu0_;
}

// An annular ring of current density.  The ring has two axis end
// points, inner and outer radii, and a constant current in Amperes.
void current_ring(const Vector &x, Vector &j)
{
   MFEM_ASSERT(x.Size() == 3, "current_ring source requires 3D space.");

   j.SetSize(x.Size());
   j = 0.0;

   Vector  a(x.Size());  // Normalized Axis vector
   Vector xu(x.Size());  // x vector relative to the axis end-point
   Vector ju(x.Size());  // Unit vector in direction of current

   xu = x;

   for (int i=0; i<x.Size(); i++)
   {
      xu[i] -= cr_params_[i];
      a[i]   = cr_params_[x.Size()+i] - cr_params_[i];
   }

   double h = a.Norml2();

   if ( h == 0.0 )
   {
      return;
   }

   double ra = cr_params_[2*x.Size()+0];
   double rb = cr_params_[2*x.Size()+1];
   if ( ra > rb )
   {
      double rc = ra;
      ra = rb;
      rb = rc;
   }
   double xa = xu*a;

   if ( h > 0.0 )
   {
      xu.Add(-xa/(h*h),a);
   }

   double xp = xu.Norml2();

   if ( xa >= 0.0 && xa <= h*h && xp >= ra && xp <= rb )
   {
      ju(0) = a(1) * xu(2) - a(2) * xu(1);
      ju(1) = a(2) * xu(0) - a(0) * xu(2);
      ju(2) = a(0) * xu(1) - a(1) * xu(0);
      ju /= h;

      j.Add(cr_params_[2*x.Size()+2]/(h*(rb-ra)),ju);
   }
}

// A Cylindrical Rod of constant magnetization.  The cylinder has two
// axis end points, a radius, and a constant magnetic field oriented
// along the axis.
void bar_magnet(const Vector &x, Vector &m)
{
   m.SetSize(x.Size());
   m = 0.0;

   Vector  a(x.Size());  // Normalized Axis vector
   Vector xu(x.Size());  // x vector relative to the axis end-point

   xu = x;

   for (int i=0; i<x.Size(); i++)
   {
      xu[i] -= bm_params_[i];
      a[i]   = bm_params_[x.Size()+i] - bm_params_[i];
   }

   double h = a.Norml2();

   if ( h == 0.0 )
   {
      return;
   }

   double  r = bm_params_[2*x.Size()];
   double xa = xu*a;

   if ( h > 0.0 )
   {
      xu.Add(-xa/(h*h),a);
   }

   double xp = xu.Norml2();

   if ( xa >= 0.0 && xa <= h*h && xp <= r )
   {
      m.Add(bm_params_[2*x.Size()+1]/h,a);
   }
}

// A Square Rod of rotating magnetized segments.  The rod is defined
// by a bounding box and a number of segments.  The magnetization in
// each segment is constant and follows a rotating pattern.
void halbach_array(const Vector &x, Vector &m)
{
   m.SetSize(x.Size());
   m = 0.0;

   // Check Bounding Box
   if ( x[0] < ha_params_[0] || x[0] > ha_params_[3] ||
        x[1] < ha_params_[1] || x[1] > ha_params_[4] ||
        x[2] < ha_params_[2] || x[2] > ha_params_[5] )
   {
      return;
   }

   int ai = (int)ha_params_[6];
   int ri = (int)ha_params_[7];
   int n  = (int)ha_params_[8];

   int i = (int)n * (x[ai] - ha_params_[ai]) /
           (ha_params_[ai+3] - ha_params_[ai]);

   m[(ri + 1 + (i % 2)) % 3] = pow(-1.0,i/2);
}

// To produce a uniform magnetic flux the vector potential can be set
// to ( By z, Bz x, Bx y).
void a_bc_uniform(const Vector & x, Vector & a)
{
   a.SetSize(3);
   a(0) = b_uniform_(1) * x(2);
   a(1) = b_uniform_(2) * x(0);
   a(2) = b_uniform_(0) * x(1);
}

// To produce a uniform magnetic field the scalar potential can be set
// to -z (or -y in 2D).
double phi_m_bc_uniform(const Vector &x)
{
   return -x(x.Size()-1);
}
