#include <igl/unproject_onto_mesh.h>
#include <igl/read_triangle_mesh.h>
#include <igl/triangle_triangle_adjacency.h>
#include <igl/barycenter.h>
#include <igl/viewer/Viewer.h>
#include "gco-v3/GCoptimization.h"
#include <iostream>


// TODO:
//  - select everything under piecewise linear path of cursor not just points
//  

int main(int argc, char *argv[])
{
  // Mesh with per-face color
  Eigen::MatrixXd V, C;
  Eigen::MatrixXi F;

  // Load a mesh in OFF format
  igl::read_triangle_mesh(argv[1],V,F);

  Eigen::MatrixXd BC;
  igl::barycenter(V,F,BC);
  Eigen::VectorXd X = 
    (BC.col(0).array()-BC.col(0).minCoeff())/
    (BC.col(0).maxCoeff()-BC.col(0).minCoeff());
  
  // Number of labels
  int K = 2;
  // per-face-per-label data term: GCO is expecting Row-major ordering
  typedef Eigen::Matrix<int,Eigen::Dynamic,Eigen::Dynamic,Eigen::RowMajor> 
    MatrixXRi;
  MatrixXRi data(F.rows(),K);
  // Phony data
  for(int f = 0;f<F.rows();f++)
  {
    data(f,0) =       X(f)*0.1*GCO_MAX_ENERGYTERM;
    data(f,1) = (1.0-X(f))*0.1*GCO_MAX_ENERGYTERM;
  }
  //
  // per-label-per-label smoothness costs
  MatrixXRi smooth = MatrixXRi::Zero(K,K);
  // Set off diagonals to punish switching labels
  smooth(0,1) = 1*GCO_MAX_ENERGYTERM;
  smooth(1,0) = 1*GCO_MAX_ENERGYTERM;

  Eigen::VectorXi L(F.rows());
  try
  {
    GCoptimizationGeneralGraph *gc = new GCoptimizationGeneralGraph(F.rows(),K);
    gc->setDataCost(data.data());
    gc->setSmoothCost(smooth.data());

    // Triangle-triangle adjacency matrix
    std::vector<std::vector<std::vector<int > > > TT;
    igl::triangle_triangle_adjacency(F,TT);
    // For each triangle
    for(int f = 0;f<TT.size();f++)
    {
      // For each edge of this triangle
      for(int ei = 0; ei<TT[f].size();ei++)
      {
        // For each neighboring triangle
        for(int ni = 0;ni<TT[f][ei].size();ni++)
        {
          // index of neighboring triangle
          int g = TT[f][ei][ni];
          gc->setNeighbors(f,g,1);
        }
      }
    }
    printf("Before optimization energy is %d\n",gc->compute_energy());
    gc->expansion(2);// run expansion for 2 iterations. For swap use gc->swap(num_iterations);
    printf("After optimization energy is %d\n",gc->compute_energy());

    // Flip data
    for(int f = 0;f<F.rows();f++)
    {
      data(f,1) =       X(f)*0.1*GCO_MAX_ENERGYTERM;
      data(f,0) = (1.0-X(f))*0.1*GCO_MAX_ENERGYTERM;
    }

    printf("Before optimization energy is %d\n",gc->compute_energy());
    gc->swap(2);// run expansion for 2 iterations. For swap use gc->swap(num_iterations);
    gc->expansion(2);// run expansion for 2 iterations. For swap use gc->swap(num_iterations);
    printf("After optimization energy is %d\n",gc->compute_energy());

    for(int f = 0;f<F.rows();f++)
    {
      L(f) = gc->whatLabel(f);
    }

    delete gc;
  } catch (GCException e)
  {
    e.Report();
  }


//  // Initialize white
//  C = Eigen::MatrixXd::Constant(F.rows(),3,1);
  igl::viewer::Viewer viewer;
//
//  const auto shoot = [&V,&F,&C,&viewer]()->bool
//  {
//    int fid;
//    Eigen::Vector3f bc;
//    // Cast a ray in the view direction starting from the mouse position
//    double x = viewer.current_mouse_x;
//    double y = viewer.core.viewport(3) - viewer.current_mouse_y;
//    if(igl::unproject_onto_mesh(Eigen::Vector2f(x,y), viewer.core.view * viewer.core.model,
//      viewer.core.proj, viewer.core.viewport, V, F, fid, bc))
//    {
//      // paint hit red
//      C.row(fid)<<1,0,0;
//      viewer.data.set_colors(C);
//      return true;
//    }
//    return false;
//  };
//  bool is_dragging_on_mesh = false;
//  viewer.callback_mouse_down = 
//    [&V,&F,&C,&shoot,&is_dragging_on_mesh]
//    (igl::viewer::Viewer& viewer, int, int)->bool
//  {
//    is_dragging_on_mesh = shoot();
//    return is_dragging_on_mesh;
//  };
//  viewer.callback_mouse_up = 
//    [&is_dragging_on_mesh] (igl::viewer::Viewer& viewer, int, int)->bool
//  {
//    is_dragging_on_mesh = false;
//    return false;
//  };
//  viewer.callback_mouse_move= 
//    [&V,&F,&C,&shoot,&is_dragging_on_mesh]
//    (igl::viewer::Viewer& viewer, int, int)->bool
//  {
//    if(is_dragging_on_mesh)
//    {
//      shoot();
//      return true;
//    }
//    return false;
//  };
//
//  std::cout<<R"(Usage:
//  [drag on mesh]  select faces
//
//)";
  // Show mesh
  viewer.data.set_mesh(V, F);
  viewer.data.set_colors(L.cast<double>());
  viewer.core.show_lines = false;
  viewer.launch();

}
