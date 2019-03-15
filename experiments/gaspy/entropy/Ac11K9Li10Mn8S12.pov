#include "colors.inc"
#include "finish.inc"

global_settings {assumed_gamma 1 max_trace_level 6}
background {color White}
camera {ultra_wide_angle
  right -108.10*x up 58.91*y
  direction 50.00*z
  location <0,0,50.00> look_at <0,0,0>}
light_source {<  2.00,   3.00,  40.00> color White
  area_light <0.70, 0, 0>, <0, 0.70, 0>, 3, 3
  adaptive 1 jitter}

#declare simple = finish {phong 0.7}
#declare pale = finish {ambient .5 diffuse .85 roughness .001 specular 0.200 }
#declare intermediate = finish {ambient 0.3 diffuse 0.6 specular 0.10 roughness 0.04 }
#declare vmd = finish {ambient .0 diffuse .65 phong 0.1 phong_size 40. specular 0.500 }
#declare jmol = finish {ambient .2 diffuse .6 specular 1 roughness .001 metallic}
#declare ase2 = finish {ambient 0.05 brilliance 3 diffuse 0.6 metallic specular 0.70 roughness 0.04 reflection 0.15}
#declare ase3 = finish {ambient .15 brilliance 2 diffuse .6 metallic specular 1. roughness .001 reflection .0}
#declare glass = finish {ambient .05 diffuse .3 specular 1. roughness .001}
#declare glass2 = finish {ambient .0 diffuse .3 specular 1. reflection .25 roughness .001}
#declare Rcell = 0.070;
#declare Rbond = 0.100;

#macro atom(LOC, R, COL, TRANS, FIN)
  sphere{LOC, R texture{pigment{color COL transmit TRANS} finish{FIN}}}
#end
#macro constrain(LOC, R, COL, TRANS FIN)
union{torus{R, Rcell rotate 45*z texture{pigment{color COL transmit TRANS} finish{FIN}}}
      torus{R, Rcell rotate -45*z texture{pigment{color COL transmit TRANS} finish{FIN}}}
      translate LOC}
#end

atom(< -6.09,  13.68, -10.05>, 1.24, rgb <0.61, 0.47, 0.78>, 0.0, ase2) // #0 
atom(< 10.02, -18.49,  -9.67>, 1.24, rgb <0.61, 0.47, 0.78>, 0.0, ase2) // #1 
atom(< -4.42, -12.10,  -6.96>, 1.91, rgb <0.44, 0.67, 0.98>, 0.0, ase2) // #2 
atom(<  7.89, -17.64,  -6.59>, 0.93, rgb <1.00, 1.00, 0.18>, 0.0, ase2) // #3 
atom(< -1.77,  13.02,  -7.27>, 1.91, rgb <0.44, 0.67, 0.98>, 0.0, ase2) // #4 
atom(< 16.43, -10.84,  -4.36>, 0.93, rgb <1.00, 1.00, 0.18>, 0.0, ase2) // #5 
atom(<-14.90,  -2.03,  -8.11>, 1.14, rgb <0.80, 0.50, 1.00>, 0.0, ase2) // #6 
atom(< -2.77,   8.38, -13.92>, 0.93, rgb <1.00, 1.00, 0.18>, 0.0, ase2) // #7 
atom(<-21.13, -15.21, -13.47>, 1.91, rgb <0.44, 0.67, 0.98>, 0.0, ase2) // #8 
atom(<  6.78, -16.86,  -7.12>, 1.14, rgb <0.80, 0.50, 1.00>, 0.0, ase2) // #9 
atom(<-19.00,  -4.31, -12.86>, 1.91, rgb <0.44, 0.67, 0.98>, 0.0, ase2) // #10 
atom(<  1.18,   2.29, -21.56>, 0.93, rgb <1.00, 1.00, 0.18>, 0.0, ase2) // #11 
atom(< -7.80, -13.98, -13.97>, 1.91, rgb <0.44, 0.67, 0.98>, 0.0, ase2) // #12 
atom(<-20.95,   5.54, -11.95>, 1.91, rgb <0.44, 0.67, 0.98>, 0.0, ase2) // #13 
atom(<-17.14, -14.24,  -8.50>, 0.93, rgb <1.00, 1.00, 0.18>, 0.0, ase2) // #14 
atom(< -5.99,   6.50,  -8.53>, 1.14, rgb <0.80, 0.50, 1.00>, 0.0, ase2) // #15 
atom(< 19.08,  -0.40,  -0.84>, 0.93, rgb <1.00, 1.00, 0.18>, 0.0, ase2) // #16 
atom(< 11.62,  -1.87, -11.97>, 1.81, rgb <0.56, 0.25, 0.83>, 0.0, ase2) // #17 
atom(<-20.27,   3.71, -13.20>, 1.14, rgb <0.80, 0.50, 1.00>, 0.0, ase2) // #18 
atom(< 12.26,   6.67,  -8.13>, 1.81, rgb <0.56, 0.25, 0.83>, 0.0, ase2) // #19 
atom(<  5.09,   3.21, -18.35>, 0.93, rgb <1.00, 1.00, 0.18>, 0.0, ase2) // #20 
atom(<  4.01,  13.95,  -6.71>, 1.81, rgb <0.56, 0.25, 0.83>, 0.0, ase2) // #21 
atom(<-18.55,   2.95, -19.40>, 1.81, rgb <0.56, 0.25, 0.83>, 0.0, ase2) // #22 
atom(< -1.62,   6.81, -16.08>, 1.91, rgb <0.44, 0.67, 0.98>, 0.0, ase2) // #23 
atom(< -0.91,  -8.48, -20.82>, 1.91, rgb <0.44, 0.67, 0.98>, 0.0, ase2) // #24 
atom(<-17.09,   2.45, -20.37>, 1.81, rgb <0.56, 0.25, 0.83>, 0.0, ase2) // #25 
atom(< -8.49,   8.57, -12.53>, 1.24, rgb <0.61, 0.47, 0.78>, 0.0, ase2) // #26 
atom(<  1.21,   5.87, -17.54>, 1.24, rgb <0.61, 0.47, 0.78>, 0.0, ase2) // #27 
atom(<-17.22,   2.62, -18.65>, 1.14, rgb <0.80, 0.50, 1.00>, 0.0, ase2) // #28 
atom(<  8.61, -19.25, -10.07>, 1.14, rgb <0.80, 0.50, 1.00>, 0.0, ase2) // #29 
atom(<-20.49,   5.26, -13.40>, 0.93, rgb <1.00, 1.00, 0.18>, 0.0, ase2) // #30 
atom(<  6.62,   4.06, -19.69>, 1.24, rgb <0.61, 0.47, 0.78>, 0.0, ase2) // #31 
atom(< -6.50, -11.98,  -6.34>, 1.24, rgb <0.61, 0.47, 0.78>, 0.0, ase2) // #32 
atom(<  0.19,  -2.57,  -6.20>, 1.91, rgb <0.44, 0.67, 0.98>, 0.0, ase2) // #33 
atom(< 15.01, -13.12,  -1.17>, 1.14, rgb <0.80, 0.50, 1.00>, 0.0, ase2) // #34 
atom(< -2.72, -11.00, -11.16>, 1.24, rgb <0.61, 0.47, 0.78>, 0.0, ase2) // #35 
atom(<-16.92,   2.85, -20.25>, 1.81, rgb <0.56, 0.25, 0.83>, 0.0, ase2) // #36 
atom(< 17.11,  -8.59,  -3.40>, 1.14, rgb <0.80, 0.50, 1.00>, 0.0, ase2) // #37 
atom(<-15.94,   2.34, -20.94>, 1.81, rgb <0.56, 0.25, 0.83>, 0.0, ase2) // #38 
atom(< 20.91,   3.14,  -0.78>, 1.91, rgb <0.44, 0.67, 0.98>, 0.0, ase2) // #39 
atom(<  8.18,   4.37, -12.54>, 1.91, rgb <0.44, 0.67, 0.98>, 0.0, ase2) // #40 
atom(<-20.21,   2.27, -24.35>, 0.93, rgb <1.00, 1.00, 0.18>, 0.0, ase2) // #41 
atom(< 12.05,   8.77, -14.46>, 1.81, rgb <0.56, 0.25, 0.83>, 0.0, ase2) // #42 
atom(< 16.41,  -5.90,   0.00>, 1.14, rgb <0.80, 0.50, 1.00>, 0.0, ase2) // #43 
atom(< 23.86,   2.20, -12.69>, 0.93, rgb <1.00, 1.00, 0.18>, 0.0, ase2) // #44 
atom(< 18.40,   6.69, -16.20>, 1.81, rgb <0.56, 0.25, 0.83>, 0.0, ase2) // #45 
atom(<-16.41,  -1.48,  -8.34>, 0.93, rgb <1.00, 1.00, 0.18>, 0.0, ase2) // #46 
atom(< -4.05,   6.12, -22.47>, 1.24, rgb <0.61, 0.47, 0.78>, 0.0, ase2) // #47 
atom(<  5.97,   2.58, -17.26>, 0.93, rgb <1.00, 1.00, 0.18>, 0.0, ase2) // #48 
atom(< 14.59, -17.54, -12.52>, 1.14, rgb <0.80, 0.50, 1.00>, 0.0, ase2) // #49 
