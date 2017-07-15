#include "gco-v3/GCoptimization.h"
#include <igl/unproject_onto_mesh.h>
#include <igl/read_triangle_mesh.h>
#include <igl/triangle_triangle_adjacency.h>
#include <igl/barycenter.h>
#include <igl/viewer/Viewer.h>
#include <GLFW/glfw3.h>
#include <thread>
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
    data(f,0) = 0;
    data(f,1) = 0.00005*GCO_MAX_ENERGYTERM;
  }
  //
  // per-label-per-label smoothness costs
  MatrixXRi smooth = MatrixXRi::Zero(K,K);
  // Set off diagonals to punish switching labels
  smooth(0,1) = 0.002*GCO_MAX_ENERGYTERM;
  smooth(1,0) = 0.002*GCO_MAX_ENERGYTERM;

  // Labels
  Eigen::VectorXi L(F.rows());
  // User constraints
  Eigen::VectorXi U = Eigen::VectorXi::Constant(F.rows(),1,-1);
  GCoptimizationGeneralGraph gc(F.rows(),K);
  gc.setDataCost(data.data());
  gc.setSmoothCost(smooth.data());

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
        gc.setNeighbors(f,g,1);
      }
    }
  }


  igl::viewer::Viewer viewer;
  viewer.data.set_mesh(V, F);
  viewer.core.show_lines = false;

  const auto update_colors = [&L,&U,&viewer]()
  {
    for(int f = 0;f<L.size();f++)
    {
      if(U(f) >= 0)
      {
        L(f) = U(f);
      }
    }
    viewer.data.set_colors(L.cast<double>());
  };
  const auto update_cut = [&gc,&update_colors,&L]()
  {
    gc.swap(1);
    for(int f = 0;f<L.size();f++)
    {
      L(f) = gc.whatLabel(f);
    }
    update_colors();
  };
  update_cut();

  std::mutex mu_loop,mu_cond;
  std::condition_variable conditional;
  bool background_thread_is_looping = true;
  bool needs_update = false;
  const auto background_loop = 
    [&background_thread_is_looping,&mu_loop,&mu_cond,&needs_update,&conditional,&update_cut]()
  {
    while(true)
    {
      {
        std::unique_lock<std::mutex> lock(mu_cond);
        conditional.wait(lock,[&needs_update](){return needs_update;});
        needs_update = false;
      }
      bool keep_going = false;
      {
        std::unique_lock<std::mutex> lock(mu_loop);
        keep_going = background_thread_is_looping;
      }
      if(!keep_going)
      {
        break;
      }
      // Call these after break so not called on finally update before exitting
      // loop
      update_cut();
      glfwPostEmptyEvent();
    }
  };
  std::thread background_thread(background_loop);

  const auto shoot = [&update_colors,&V,&F,&C,&viewer,&gc,&data,&L,&U,&mu_cond,&conditional,&needs_update]()->bool
  {
    int fid;
    Eigen::Vector3f bc;
    // Cast a ray in the view direction starting from the mouse position
    double x = viewer.current_mouse_x;
    double y = viewer.core.viewport(3) - viewer.current_mouse_y;
    if(igl::unproject_onto_mesh(Eigen::Vector2f(x,y), viewer.core.view * viewer.core.model,
      viewer.core.proj, viewer.core.viewport, V, F, fid, bc))
    {
      L(fid) = 1;
      U(fid) = 1;
      data(fid,0) = GCO_MAX_ENERGYTERM;
      data(fid,1) = 0;

      {
        std::lock_guard<std::mutex> lock(mu_cond);
        needs_update = true;
      }
      conditional.notify_all();

      update_colors();
      return true;
    }
    return false;
  };

  bool is_dragging_on_mesh = false;
  viewer.callback_mouse_down = 
    [&V,&F,&C,&shoot,&update_cut,&is_dragging_on_mesh,&gc]
    (igl::viewer::Viewer& viewer, int, int)->bool
  {
    is_dragging_on_mesh = shoot();
    return is_dragging_on_mesh;
  };
  viewer.callback_mouse_up = 
    [&is_dragging_on_mesh,&update_cut,&gc,&L] 
    (igl::viewer::Viewer& viewer, int, int)->bool
  {
    //if(is_dragging_on_mesh)
    //{
    //  update_cut();
    //}
    is_dragging_on_mesh = false;
    return false;
  };
  viewer.callback_mouse_move= 
    [&V,&F,&C,&shoot,&update_cut,&is_dragging_on_mesh,&gc]
    (igl::viewer::Viewer& viewer, int, int)->bool
  {
    if(is_dragging_on_mesh)
    {
      shoot();
      return true;
    }
    return false;
  };
  viewer.callback_key_down =
    [&](igl::viewer::Viewer & viewer, unsigned char key, int mod)->bool
   {
    switch(key)
    {
      default:
        return false;
      case 'L':
      {
        std::cout<<L.transpose().cast<double>()<<std::endl;
        break;
      }
    }
   };


  std::cout<<R"(Usage:
  [drag on mesh]  select faces

)";
  // Show mesh
  viewer.launch();
  {
    std::lock_guard<std::mutex> lock(mu_loop);
    background_thread_is_looping = false;
  }
  {
    std::lock_guard<std::mutex> lock(mu_cond);
    needs_update = true;
  }
  conditional.notify_all();

  background_thread.join();

}
