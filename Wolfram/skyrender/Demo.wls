#!/usr/bin/env wolframscript
(* ::Package:: *)

(* SkyRender Demo
   ------------------------------------------------------------------
   Run locally:
       wolframscript -file Demo.wls
   This renders a 256x144 v=0.95 Alcubierre frame using a procedural
   starfield and writes it to skyrender_demo.png (1-2 minutes on a
   modern laptop CPU).

   To run the same calculation on a Wolfram Cloud GPU instead, set
   USE_GPU = True below.  You'll need an entitled Wolfram account
   and a sky-image path that's reachable by the remote runtime
   (RemoteInputFiles uploads the file).
*)

USE_GPU = False;
SKY_PATH = ""; (* e.g. "/Users/me/milky_way_panorama.jpg" — leave empty to use procedural sky *)

Get[FileNameJoin[{DirectoryName[$InputFileName], "SkyRender.wl"}]];

If[USE_GPU,
  Print["Submitting to Wolfram Cloud GPU (L40S)…"];
  job = SubmitFrameToGPU[
    "alcubierre",
    <|"v0" -> 0.95, "R" -> 1.0, "sigma" -> 8.0|>,
    SKY_PATH,
    {"Width" -> 1280, "Height" -> 720, "NSteps" -> 280, "Dlam" -> 0.15}
  ];
  Print["Job submitted: ", job];
  Print["Status: ", job["Status"]];
  Print["Once Status -> \"Completed\", get the image with: img = job[\"Result\"]"];
  ,
  Print["Local CPU render…"];
  sky = If[SKY_PATH =!= "" && FileExistsQ[SKY_PATH],
    LoadEquirectangularSky[SKY_PATH, "Gain" -> 1.6],
    ProceduralSky[2026, 4000]
  ];
  img = RenderFrame[
    "alcubierre",
    <|"v0" -> 0.95, "R" -> 1.0, "sigma" -> 8.0|>,
    sky,
    {"Width" -> 256, "Height" -> 144, "NSteps" -> 200, "Dlam" -> 0.15}
  ];
  Export[FileNameJoin[{DirectoryName[$InputFileName], "skyrender_demo.png"}], img];
  Print["Wrote skyrender_demo.png (", ImageDimensions[img], ")"]
]
