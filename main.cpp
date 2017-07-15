#include <igl/unproject_onto_mesh.h>
#include <igl/read_triangle_mesh.h>
#include <igl/viewer/Viewer.h>
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

  // Initialize white
  C = Eigen::MatrixXd::Constant(F.rows(),3,1);
  igl::viewer::Viewer viewer;

  const auto shoot = [&V,&F,&C,&viewer]()->bool
  {
    int fid;
    Eigen::Vector3f bc;
    // Cast a ray in the view direction starting from the mouse position
    double x = viewer.current_mouse_x;
    double y = viewer.core.viewport(3) - viewer.current_mouse_y;
    if(igl::unproject_onto_mesh(Eigen::Vector2f(x,y), viewer.core.view * viewer.core.model,
      viewer.core.proj, viewer.core.viewport, V, F, fid, bc))
    {
      // paint hit red
      C.row(fid)<<1,0,0;
      viewer.data.set_colors(C);
      return true;
    }
    return false;
  };
  bool is_dragging_on_mesh = false;
  viewer.callback_mouse_down = 
    [&V,&F,&C,&shoot,&is_dragging_on_mesh]
    (igl::viewer::Viewer& viewer, int, int)->bool
  {
    is_dragging_on_mesh = shoot();
    return is_dragging_on_mesh;
  };
  viewer.callback_mouse_up = 
    [&is_dragging_on_mesh] (igl::viewer::Viewer& viewer, int, int)->bool
  {
    is_dragging_on_mesh = false;
    return false;
  };
  viewer.callback_mouse_move= 
    [&V,&F,&C,&shoot,&is_dragging_on_mesh]
    (igl::viewer::Viewer& viewer, int, int)->bool
  {
    if(is_dragging_on_mesh)
    {
      shoot();
      return true;
    }
    return false;
  };

  std::cout<<R"(Usage:
  [drag on mesh]  select faces

)";
  // Show mesh
  viewer.data.set_mesh(V, F);
  viewer.data.set_colors(C);
  viewer.core.show_lines = false;
  viewer.launch();
}
