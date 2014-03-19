//---------------------------------------------------------------------------//
// Copyright (c) 2013 Kyle Lutz <kyle.r.lutz@gmail.com>
//
// Distributed under the Boost Software License, Version 1.0
// See accompanying file LICENSE_1_0.txt or copy at
// http://www.boost.org/LICENSE_1_0.txt
//
// See http://kylelutz.github.com/compute for more information.
//---------------------------------------------------------------------------//

#include <GL/glew.h>
#if defined (__APPLE__) || defined(MACOSX)
    #include <OpenGL/OpenGL.h>
    #include <GLUT/glut.h>
#else
    #include <GL/freeglut.h>
    #ifdef UNIX
       #include <GL/glx.h>
    #endif
#endif

#include <boost/compute/source.hpp>
#include <boost/compute/interop/opengl/context.hpp>
#include <boost/compute/interop/opengl/opengl_buffer.hpp>
#include <boost/compute/interop/opengl.hpp>
#include <boost/compute/command_queue.hpp>
#include <boost/compute/program.hpp>
#include <boost/compute/buffer.hpp>
#include <boost/compute/kernel.hpp>
namespace compute = boost::compute;

// Constants, defines, typedefs and global declarations
//*****************************************************************************
#define REFRESH_DELAY	  10 //ms

// vbo variables
GLuint vbo;
int iGLUTWindowHandle = 0;          // handle to the GLUT window

// mouse controls
int mouse_old_x, mouse_old_y;
int mouse_buttons = 0;
float rotate_x = 0.0, rotate_y = 0.0;
float translate_z = -3.0;

// Rendering window vars
float anim = 0.0;

const unsigned int window_width = 512;
const unsigned int window_height = 512;
const unsigned int mesh_width = 256;
const unsigned int mesh_height = 256;
size_t szGlobalWorkSize[] = {mesh_width, mesh_height};

boost::compute::context cxt;
boost::compute::device dev;
boost::compute::command_queue queue;
boost::compute::program prog;
boost::compute::kernel ker;
boost::compute::opengl_buffer opengl_buf;


void cleanup(int exitCode)
{
   if(vbo)
   {
      glBindBuffer(1, vbo);
      glDeleteBuffers(1, &vbo);
      vbo = 0;
   }

   exit(exitCode);
}


void runKernel()
{
   // acqurire buffer so that it is accessible to OpenCL
   boost::compute::opengl_enqueue_acquire_buffer(opengl_buf, queue);

   //set another kernel argument
   ker.set_arg<compute::float_>(3,anim);

   //execute the kernel
   size_t offset[2] = { 0, 0 };
   queue.enqueue_nd_range_kernel(ker, 2, offset,szGlobalWorkSize);
   queue.finish();

   // release buffer so that it is accessible to OpenGL
   boost::compute::opengl_enqueue_release_buffer(opengl_buf, queue);   
}

void timerEvent(int value)
{
   glutPostRedisplay();
   glutTimerFunc(REFRESH_DELAY, timerEvent,0);
}

void DisplayGL()
{
   anim += 0.01f;

   // run OpenCL kernel to generate vertex positions
   runKernel();

   
   // clear graphics then render from the vbo
   glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
   glBindBuffer(GL_ARRAY_BUFFER, vbo);
   glVertexPointer(4, GL_FLOAT, 0, 0);
   glEnableClientState(GL_VERTEX_ARRAY);
   glColor3f(1.0, 0.0, 0.0);
   glDrawArrays(GL_POINTS, 0, mesh_width * mesh_height);
   glDisableClientState(GL_VERTEX_ARRAY);
   
   // flip backbuffer to screen
   glutSwapBuffers();
}

void KeyboardGL(unsigned char key, int x, int y)
{
   switch(key) 
   {
      case '\033': // escape quits
      case '\015': // Enter quits    
      case 'Q':    // Q quits
      case 'q':    // q (or escape) quits
	 // Cleanup up and quit	 
	 cleanup(EXIT_SUCCESS);
	 break;
    }
}

void mouse(int button, int state, int x, int y)
{
   if (state == GLUT_DOWN)
   {
      mouse_buttons |= 1<<button;
   } else if (state == GLUT_UP)
   {
      mouse_buttons = 0;
   }

    mouse_old_x = x;
    mouse_old_y = y;
}

void motion(int x, int y)
{
    float dx, dy;
    dx = (float)(x - mouse_old_x);
    dy = (float)(y - mouse_old_y);

    if (mouse_buttons & 1) {
        rotate_x += dy * 0.2f;
        rotate_y += dx * 0.2f;
    } else if (mouse_buttons & 4) {
        translate_z += dy * 0.01f;
    }

    mouse_old_x = x;
    mouse_old_y = y;

    // set view matrix
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
    glTranslatef(0.0, 0.0, translate_z);
    glRotatef(rotate_x, 1.0, 0.0, 0.0);
    glRotatef(rotate_y, 0.0, 1.0, 0.0);
}


bool initCL()
{
   cxt = boost::compute::opengl_create_shared_context();

   dev = cxt.get_device();


   queue = boost::compute::command_queue(cxt,dev);

   if(cxt == NULL || queue == NULL)
      return false;
   else
      return true;
}

void InitGL(int* argc, char** argv)
{
   // initialize GLUT 
   glutInit(argc, argv);
   glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE);
   glutInitWindowPosition (glutGet(GLUT_SCREEN_WIDTH)/2 - window_width/2, 
			   glutGet(GLUT_SCREEN_HEIGHT)/2 - window_height/2);
   glutInitWindowSize(window_width, window_height);
   iGLUTWindowHandle = glutCreateWindow("OpenCL/GL Interop (VBO)");
#if !(defined (__APPLE__) || defined(MACOSX))
   glutSetOption(GLUT_ACTION_ON_WINDOW_CLOSE, GLUT_ACTION_GLUTMAINLOOP_RETURNS);
#endif
   
   // register GLUT callback functions
   glutDisplayFunc(DisplayGL);
   glutKeyboardFunc(KeyboardGL);
   glutMouseFunc(mouse);
   glutMotionFunc(motion);
   glutTimerFunc(REFRESH_DELAY, timerEvent,0);
   
   // initialize necessary OpenGL extensions
   glewInit();
   GLboolean bGLEW = glewIsSupported("GL_VERSION_2_0 GL_ARB_pixel_buffer_object"); 
   //shrCheckErrorEX(bGLEW, shrTRUE, pCleanup);
   
   // default initialization
   glClearColor(0.0, 0.0, 0.0, 1.0);
   glDisable(GL_DEPTH_TEST);
   
   // viewport
   glViewport(0, 0, window_width, window_height);
   
   // projection
   glMatrixMode(GL_PROJECTION);
   glLoadIdentity();
   gluPerspective(60.0, (GLfloat)window_width / (GLfloat) window_height, 0.1, 10.0);
   
   // set view matrix
   glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
   glMatrixMode(GL_MODELVIEW);
   glLoadIdentity();
   glTranslatef(0.0, 0.0, translate_z);
   glRotatef(rotate_x, 1.0, 0.0, 0.0);
   glRotatef(rotate_y, 0.0, 1.0, 0.0);
   
   return;
}

const char source[] = BOOST_COMPUTE_STRINGIZE_SOURCE(
   __kernel void sine_wave(__global float4* pos, unsigned int width, unsigned int height, float time)
   {
      unsigned int x = get_global_id(0);
      unsigned int y = get_global_id(1);
      
      // calculate uv coordinates
      float u = x / (float) width;
      float v = y / (float) height;
      u = u*2.0f - 1.0f;
      v = v*2.0f - 1.0f;
      
      // calculate simple sine wave pattern
      float freq = 4.0f;
      float w = sin(u*freq + time) * cos(v*freq + time) * 0.5f;
      
      // write output vertex
      pos[y*width+x] = (float4)(u, w, v, 1.0f);
   }
   );


// Create VBO
//*****************************************************************************
void createVBO(GLuint* vbo)
{
   // create VBO
   unsigned int size = mesh_width * mesh_height * 4 * sizeof(float);
      
   // create buffer object
   glGenBuffers(1, vbo);
   glBindBuffer(GL_ARRAY_BUFFER, *vbo);
   
   // initialize buffer object
   glBufferData(GL_ARRAY_BUFFER, size, 0, GL_DYNAMIC_DRAW);
   
   // create OpenCL buffer from GL VBO
   opengl_buf = boost::compute::opengl_buffer(cxt,*vbo,CL_MEM_WRITE_ONLY);   
}

int main(int argc, char** argv)
{
   InitGL(&argc, argv);

   if(!initCL())
      exit(EXIT_FAILURE);

   prog = boost::compute::program::create_with_source(source,cxt);

   //build the program
   prog.build();


   // create VBO (if using standard GL or CL-GL interop), otherwise create Cl buffer
   createVBO(&vbo);

   //now setup the kernel
   ker = boost::compute::kernel(prog,"sine_wave");
   ker.set_arg(0,opengl_buf);
   ker.set_arg<compute::uint_>(1,mesh_width);
   ker.set_arg<compute::uint_>(2,mesh_height);

   glutMainLoop();
   
   return 0;
}
