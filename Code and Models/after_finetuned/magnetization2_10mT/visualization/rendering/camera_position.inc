// POV-Ray 3.6 / 3.7 Scene File "Ribbon_Cable_1.pov"
// author: Friedrich A. Lohmueller, Sept-2009/Jan-2011
// email: Friedrich.Lohmueller_at_t-online.de
// homepage: http://www.f-lohmueller.de
//--------------------------------------------------------------------------
#version 3.6; // 3.7;
global_settings{ assumed_gamma 1.0 }
#default{ finish{ ambient 0.1 diffuse 0.9 }}
//--------------------------------------------------------------------------
#include "colors.inc"
#include "textures.inc"
#include "glass.inc"
#include "metals.inc"
#include "golds.inc"
#include "stones.inc"
#include "woods.inc"
#include "shapes.inc"
#include "shapes2.inc"
#include "functions.inc"
#include "math.inc"
#include "transforms.inc"
//--------------------------------------------------------------------------------------------------------<<<<
//------------------------------------------------------------- Camera_Position, Camera_look_at, Camera_Angle
	#declare Camera_Number = 0;
//----------------------------------------89----------------------------------------------------------------<<<<
#switch ( Camera_Number )
#case (0) // 3D reaching arm
  #declare Ultra_Wide_Angle_On = 0;// don't use fish eye - for proper text
  #declare Camera_Position = <1.85, 5.0, -10.00> ;  // Topside view
  #declare Camera_Look_At  = <1.85, -1.0, 3.0> ;
  #declare Camera_Angle    =  30 ;
  #declare Camera_Rotate = <0,0,0>; // tilling of the camera!!
#break
#else  // alternate view
  #declare Ultra_Wide_Angle_On = 0;// don't use fish eye - for proper text
  #declare Camera_Position = <23.00, 6.00,-18.00> ;  // Topside view
  #declare Camera_Look_At  = < 0.50, 1.700,  7.00> ;
  #declare Camera_Angle    =  28 ;
  #declare Camera_Rotate = <0,0,0>; // tilling of the camera!!
#break
#end // of "#switch ( Camera_Number )" -----------------------------
//---------------------------------------------------------------------

camera{ location Camera_Position
        right    x*image_width/image_height
        angle    Camera_Angle
        look_at  Camera_Look_At
      }
//light_source{< 0,300,500> color White}
light_source{< 1500,2500,-1000>color White}           // sun light
light_source{ Camera_Position  color rgb<0.9,0.9,1>*0.1}  // flash light


/*
// fog ---------------------------------------------------------------
fog{fog_type   2   distance 65  color rgb<1,1,1>
fog_offset 0.1 fog_alt  2.0 turbulence 0.2}
//--------------------------------------------------------------------
*/
// <1,0,0>, -3
// <0,1,0>, 0
//<0,0,1>,-200
background { color White }

plane { <0,1,0>, -0.131 // plane with layered textures 需要该的地方
texture { pigment{color White*1.1}
             }
}


//--------------------------------------------------------------------------
//---------------------------- objects in scene ----------------------------
//--------------------------------------------------------------------------
