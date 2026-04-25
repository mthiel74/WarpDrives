(* ::Package:: *)

(* ::Title:: *)
(*WarpBubbleSim  \[Dash]  Wolfram Language port of WarpDrives*)

(* ::Text:: *)
(*Master loader.  Call with*)
(*    Get["/absolute/path/to/WarpBubbleSim.wl"]*)
(*from any notebook and every sub-package becomes available.*)

BeginPackage["WarpBubbleSim`"];

$WarpBubbleSimRoot::usage =
  "$WarpBubbleSimRoot is the absolute directory of WarpBubbleSim.wl.";

Begin["`Private`"];

$WarpBubbleSimRoot = DirectoryName[
  If[$Input === "" || $Input === None, $InputFileName, $Input]];
If[$WarpBubbleSimRoot === "" || !DirectoryQ[$WarpBubbleSimRoot],
  $WarpBubbleSimRoot = DirectoryName[$InputFileName]];

AppendTo[$Path, $WarpBubbleSimRoot];

End[];
EndPackage[];

(* Load sub-packages in dependency order *)
Get["ShapeFunctions`"];
Get["Metrics`"];
Get["Tensors`"];
Get["ADM`"];
Get["Energy`"];
Get["Conditions`"];
Get["Invariants`"];
Get["Geodesics`"];
Get["Visualize`"];

Print["[WarpBubbleSim] Loaded packages: ShapeFunctions, Metrics, Tensors, ADM, ",
      "Energy, Conditions, Invariants, Geodesics, Visualize"];
