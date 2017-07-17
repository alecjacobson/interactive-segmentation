#include "gco-v3/GCoptimization.h"
#include <igl/unproject_onto_mesh.h>
#include <igl/read_triangle_mesh.h>
#include <igl/remove_duplicate_vertices.h>
#include <igl/triangle_triangle_adjacency.h>
#include <igl/barycenter.h>
#include <igl/per_face_normals.h>
#include <igl/parula.h>
#include <igl/edge_lengths.h>
#include <igl/viewer/Viewer.h>
#include <GLFW/glfw3.h>
#include <thread>
#include <iostream>


// TODO:
//  - select everything under piecewise linear path of cursor not just points
//  - select everything within some brush radius
//  - weight smoothness term by edge lengths
//  - weight data term by local area
//  

int main(int argc, char *argv[])
{
  Eigen::MatrixXd V;
  Eigen::MatrixXi F;

  // Load a mesh in OFF format
  igl::read_triangle_mesh(argv[1],V,F);
  {
    // Remove duplicate vertices (especially needed for .stl files)
    Eigen::VectorXi I,J;
    igl::remove_duplicate_vertices(
      Eigen::MatrixXd(V),Eigen::MatrixXi(F),0,V,I,J,F);
  }

  Eigen::MatrixXd BC;
  igl::barycenter(V,F,BC);
  Eigen::MatrixXd N,EL;
  igl::per_face_normals(V,F,N);
  igl::edge_lengths(V,F,EL);
  
  // Number of labels
  int K = 3;
  int selected_id = 1;
  // per-face-per-label data term: GCO is expecting Row-major ordering
  typedef Eigen::Matrix<int,Eigen::Dynamic,Eigen::Dynamic,Eigen::RowMajor> 
    MatrixXRi;
  MatrixXRi data(F.rows(),K);
  // Phony data
  for(int f = 0;f<F.rows();f++)
  {
    // Initialize everything to slightly lean toward id=0
    for(int j = 0;j<K;j++)
    {
      data(f,j) = j==0 ? 0 : 0.00005*GCO_MAX_ENERGYTERM;
    }
  }
  //
  // per-label-per-label smoothness costs
  MatrixXRi smooth = MatrixXRi::Constant(K,K,1);
  // Set off diagonals to punish switching labels
  smooth.diagonal().setConstant(0);

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
      const double w_el = EL(f,ei);
      // For each neighboring triangle
      for(int ni = 0;ni<TT[f][ei].size();ni++)
      {
        // index of neighboring triangle
        int g = TT[f][ei][ni];
        // "Randomized Cuts for 3D Mesh Analysis" [Golovinskiy and Funkhouser
        // 2008]
        // 
        // "if θ is the exterior dihedral angle across an edge, we define a
        // concave weight w(θ ) = min((θ /π )α , 1)"
        //
        const double alpha = 10.0;
        gc.setNeighbors(
          f,g, 
          w_el * 
          std::pow( std::abs(N.row(f).dot(N.row(g))),alpha)
          *GCO_MAX_ENERGYTERM );
      }
    }
  }


  igl::viewer::Viewer viewer;
  viewer.data.set_mesh(V, F);
  viewer.core.show_lines = false;

  std::mutex mu;
  std::condition_variable conditional;
  bool background_thread_is_looping = true;
  bool needs_update = false;

  const auto update_colors = [&L,&U,&viewer,&K]()
  {
    for(int f = 0;f<L.size();f++)
    {
      if(U(f) >= 0)
      {
        L(f) = U(f);
      }
    }
    Eigen::MatrixXd C;
    igl::parula(L,0,K-1,C);
    viewer.data.set_colors(C);
  };
  const auto update_cut = [&gc,&update_colors,&L,&mu,&K]()
  {
    try
    {
      //gc.expansion(K-1);
      gc.swap(K-1);
    }catch(GCException e)
    {
      e.Report();
    }
    for(int f = 0;f<L.size();f++)
    {
      L(f) = gc.whatLabel(f);
    }
    {
      std::lock_guard<std::mutex> lock(mu);
      update_colors();
    }
  };
  update_cut();

  const auto background_loop = 
    [&background_thread_is_looping,&mu,&needs_update,&conditional,&update_cut]()
  {
    while(true)
    {
      {
        std::unique_lock<std::mutex> lock(mu);
        conditional.wait(lock,[&needs_update](){return needs_update;});
        needs_update = false;
      }
      bool keep_going = false;
      {
        std::unique_lock<std::mutex> lock(mu);
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

  const auto shoot = 
    [&K,&selected_id,&update_colors,&V,&F,&viewer,&gc,&data,&L,&U,&conditional,&needs_update,&mu]()->bool
  {
    int fid;
    Eigen::Vector3f bc;
    // Cast a ray in the view direction starting from the mouse position
    double x = viewer.current_mouse_x;
    double y = viewer.core.viewport(3) - viewer.current_mouse_y;
    if(igl::unproject_onto_mesh(Eigen::Vector2f(x,y), viewer.core.view * viewer.core.model,
      viewer.core.proj, viewer.core.viewport, V, F, fid, bc))
    {
      L(fid) = selected_id;
      U(fid) = selected_id;
      for(int j = 0;j<K;j++)
      {
        data(fid,j) = j==selected_id ? 0 : GCO_MAX_ENERGYTERM;
      }

      {
        std::lock_guard<std::mutex> lock(mu);
        needs_update = true;
      }
      conditional.notify_all();

      {
        std::lock_guard<std::mutex> lock(mu);
        update_colors();
      }
      return true;
    }
    return false;
  };

  bool is_dragging_on_mesh = false;
  viewer.callback_mouse_down = 
    [&V,&F,&shoot,&is_dragging_on_mesh,&gc]
    (igl::viewer::Viewer& viewer, int, int)->bool
  {
    is_dragging_on_mesh = shoot();
    return is_dragging_on_mesh;
  };
  viewer.callback_mouse_up = 
    [&is_dragging_on_mesh,&gc,&L] 
    (igl::viewer::Viewer& viewer, int, int)->bool
  {
    is_dragging_on_mesh = false;
    return false;
  };
  viewer.callback_mouse_move= 
    [&V,&F,&shoot,&is_dragging_on_mesh,&gc]
    (igl::viewer::Viewer& viewer, int, int)->bool
  {
    if(is_dragging_on_mesh)
    {
      shoot();
      return true;
    }
    return false;
  };
  viewer.callback_key_pressed =
    [&](igl::viewer::Viewer & viewer, unsigned char key, int mod)->bool
   {
    switch(key)
    {
      default:
        return false;
      case '<':
      {
        selected_id = (selected_id-1+K)%K;
        break;
      }
      case '>':
      {
        selected_id = (selected_id+1)%K;
        break;
      }
      case 'L':
      {
        std::cout<<L.transpose().cast<double>()<<std::endl;
        break;
      }
    }
        return true;
   };


  std::cout<<R"(Usage:
  [drag on mesh]  select faces

)";
  // Show mesh
  viewer.launch();
  {
    std::lock_guard<std::mutex> lock(mu);
    background_thread_is_looping = false;
  }
  {
    std::lock_guard<std::mutex> lock(mu);
    needs_update = true;
  }
  conditional.notify_all();

  background_thread.join();

}
