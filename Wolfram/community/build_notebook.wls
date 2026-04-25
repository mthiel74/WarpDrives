#!/usr/bin/env wolframscript
(* ----------------------------------------------------------------
   Build Wolfram/community/warpdrives.nb -- a Wolfram Notebook for
   the Wolfram Community.

   Each section pairs an explanation with an Input cell containing the
   exact Wolfram-Language code that produces the figure or animation,
   followed by an Output cell with a pre-rendered result so the post
   reads top-to-bottom in the browser without anyone having to evaluate
   anything.  The animated GIFs are wrapped in AnimatedImage, which the
   Wolfram Community viewer preserves as a looping <img>.

   Run after build_assets.wls has produced ./results/.
   ---------------------------------------------------------------- *)

hereDir = DirectoryName[If[$ScriptCommandLine =!= {}, $InputFileName,
                           NotebookFileName[]]];
resDir  = FileNameJoin[{hereDir, "results"}];

imgWidth = 720;

(* ---------- math typesetting helpers -------------------------------- *)

mathI[box_] := Cell[BoxData[FormBox[box, TraditionalForm]],
                    "InlineFormula", FontFamily -> "Times"];
mathD[box_] := Cell[BoxData[FormBox[box, TraditionalForm]],
                    "DisplayFormula", TextAlignment -> Center,
                    FontFamily -> "Times"];

inputCell[s_String] := Cell[s, "Input", CellTags -> {"runnable"}];

(* ---------- image / animation helpers ------------------------------- *)

loadPng[name_, w_:imgWidth, caption_:""] :=
  With[{p = FileNameJoin[{resDir, name}]},
    If[!FileExistsQ[p],
      Style["[missing " <> name <> "]", Red],
      If[caption === "",
        Image[Import[p], ImageSize -> w],
        Column[{Image[Import[p], ImageSize -> w],
                Pane[Style[caption, Italic, FontSize -> 10,
                           FontFamily -> "Helvetica"],
                     w, Alignment -> Center]},
               Alignment -> Center, Spacings -> 0.6]]]];

(* AnimatedImage from a GIF.  We subsample frames so the .nb stays
   small but the animation is still smooth in the Community viewer. *)
loadAnim[name_, w_:480, frameCap_:30, rate_:6] :=
  With[{p = FileNameJoin[{resDir, name}]},
    If[!FileExistsQ[p],
      Style["[missing " <> name <> "]", Red],
      With[{all = Import[p, {"GIF", "ImageList"}]},
        With[{stride = Max[1, Ceiling[Length[all]/frameCap]]},
          AnimatedImage[
            ImageResize[#, {w}] & /@ all[[1 ;; -1 ;; stride]],
            FrameRate -> rate,
            AnimationRunning -> True,
            AnimationRepetitions -> Infinity]]]]];

imgCell[content_, caption_String:""] :=
  Cell[BoxData[ToBoxes[
    If[caption === "", content,
      Column[{content,
              Pane[Style[caption, Italic, FontSize -> 10,
                         FontFamily -> "Helvetica"],
                   imgWidth, Alignment -> Center]},
             Alignment -> Center, Spacings -> 0.6]]]],
    "Output", ShowCellLabel -> False,
    TextAlignment -> Center,
    CellMargins -> {{20, 20}, {10, 14}}];

(* ---------- math symbols ------------------------------------------ *)

vsBoxed   = SubscriptBox["v", "s"];
xsBoxed   = SubscriptBox["x", "s"];
rsBoxed   = SubscriptBox["r", "s"];
fOfRs     = RowBox[{"f", "(", rsBoxed, ")"}];
sigmaSym  = "\[Sigma]";
ggMnLow   = SubscriptBox["g", RowBox[{"\[Mu]", "\[Nu]"}]];
GMnLow    = SubscriptBox["G", RowBox[{"\[Mu]", "\[Nu]"}]];
TMnLow    = SubscriptBox["T", RowBox[{"\[Mu]", "\[Nu]"}]];
RMnLow    = SubscriptBox["R", RowBox[{"\[Mu]", "\[Nu]"}]];
RAbgdLow  = SubscriptBox["R",
              RowBox[{"\[Alpha]", "\[Beta]", "\[Gamma]", "\[Delta]"}]];
betaUp    = SuperscriptBox["\[Beta]", "i"];
gammaIJ   = SubscriptBox["\[Gamma]", RowBox[{"i", "j"}]];
alphaSym  = "\[Alpha]";
KijLow    = SubscriptBox["K", RowBox[{"i", "j"}]];

(* ---------- display equations ------------------------------------- *)

alcMetricEq = RowBox[{
  SuperscriptBox["ds", "2"], "=",
  RowBox[{"-", SuperscriptBox["dt", "2"]}], "+",
  SuperscriptBox[
    RowBox[{"(", RowBox[{"dx", "-",
      RowBox[{vsBoxed, " ", fOfRs, " ", "dt"}]}], ")"}], "2"],
  "+", SuperscriptBox["dy", "2"], "+", SuperscriptBox["dz", "2"]}];

shapeFnEq = RowBox[{"f", "(", "r", ")", "=",
  FractionBox[
    RowBox[{RowBox[{"tanh", "(",
              RowBox[{sigmaSym, "(", RowBox[{"r", "+", "R"}], ")"}], ")"}],
            "-",
            RowBox[{"tanh", "(",
              RowBox[{sigmaSym, "(", RowBox[{"r", "-", "R"}], ")"}], ")"}]}],
    RowBox[{"2", "tanh", "(", RowBox[{sigmaSym, " ", "R"}], ")"}]]}];

admEq = RowBox[{
  SuperscriptBox["ds", "2"], "=",
  RowBox[{"-", SuperscriptBox[alphaSym, "2"], " ", SuperscriptBox["dt", "2"]}],
  "+", gammaIJ,
  RowBox[{"(", RowBox[{SuperscriptBox["dx", "i"], "+",
    RowBox[{SuperscriptBox["\[Beta]", "i"], "dt"}]}], ")"}],
  RowBox[{"(", RowBox[{SuperscriptBox["dx", "j"], "+",
    RowBox[{SuperscriptBox["\[Beta]", "j"], "dt"}]}], ")"}]}];

einsteinEq = RowBox[{GMnLow, "=", RowBox[{"8", "\[Pi]", " ", TMnLow}]}];

stressEq = RowBox[{TMnLow, "=", FractionBox[GMnLow, RowBox[{"8", "\[Pi]"}]]}];

rhoAnalytic = RowBox[{"\[Rho]", "=",
  RowBox[{"-", FractionBox[SuperscriptBox[vsBoxed, "2"], RowBox[{"32", "\[Pi]"}]],
    SuperscriptBox[
      RowBox[{"(", FractionBox["df", RowBox[{"d", rsBoxed}]], ")"}], "2"],
    " ",
    FractionBox[RowBox[{SuperscriptBox["y", "2"], "+", SuperscriptBox["z", "2"]}],
                SuperscriptBox[rsBoxed, "2"]]}]}];

thetaEq = RowBox[{"\[Theta]", "=", "\[Del]", "\[CenterDot]",
  RowBox[{"\[Beta]", "=",
    FractionBox[RowBox[{vsBoxed, "(", RowBox[{"x", "-", xsBoxed}], ")"}],
                rsBoxed],
    " ", FractionBox["df", RowBox[{"d", rsBoxed}]]}]}];

geodesicEq = RowBox[{
  FractionBox[RowBox[{SuperscriptBox["d", "2"], SuperscriptBox["x", "\[Mu]"]}],
              RowBox[{"d", SuperscriptBox["\[Lambda]", "2"]}]],
  "+",
  RowBox[{SubsuperscriptBox["\[CapitalGamma]", RowBox[{"\[Alpha]", "\[Beta]"}], "\[Mu]"],
    " ",
    FractionBox[RowBox[{"d", SuperscriptBox["x", "\[Alpha]"]}], "d\[Lambda]"],
    " ",
    FractionBox[RowBox[{"d", SuperscriptBox["x", "\[Beta]"]}], "d\[Lambda]"]}],
  "=", "0"}];

wecEq = RowBox[{TMnLow, " ",
  SuperscriptBox["u", "\[Mu]"], " ", SuperscriptBox["u", "\[Nu]"],
  "\[GreaterEqual]", "0", "\[ThickSpace]\[ThickSpace]\[ThickSpace]",
  RowBox[{"\[ForAll]", " ", "timelike", " ", "u"}]}];

necEq = RowBox[{TMnLow, " ",
  SuperscriptBox["k", "\[Mu]"], " ", SuperscriptBox["k", "\[Nu]"],
  "\[GreaterEqual]", "0", "\[ThickSpace]\[ThickSpace]\[ThickSpace]",
  RowBox[{"\[ForAll]", " ", "null", " ", "k"}]}];

(* ----- Section 1 extras ----- *)

xsTrajectoryEq = RowBox[{xsBoxed, "(", "t", ")", "=",
  RowBox[{SubscriptBox["x", "0"], "+",
    RowBox[{SuperscriptBox[SubscriptBox["\[Integral]", "0"], "t"], " ",
      RowBox[{vsBoxed, "(", "t'", ")"}], " ",
      RowBox[{"d", "t'"}]}]}]}];

rsDefEq = RowBox[{rsBoxed, "=",
  SqrtBox[RowBox[{
    SuperscriptBox[RowBox[{"(", RowBox[{"x", "-", xsBoxed}], ")"}], "2"],
    "+", SuperscriptBox["y", "2"], "+", SuperscriptBox["z", "2"]}]]}];

fLimitEq = RowBox[{
  RowBox[{RowBox[{"f", "(", "0", ")"}], "=", "1"}],
  ",", "\[ThickSpace]\[ThickSpace]\[ThickSpace]",
  RowBox[{
    SubscriptBox[RowBox[{"lim"}], RowBox[{"r", "\[Rule]", "\[Infinity]"}]],
    " ", RowBox[{"f", "(", "r", ")"}]}], "=", "0"}];

(* ----- Section 2 extras (ADM / shift / lapse) ----- *)

alcShiftEq = RowBox[{
  SuperscriptBox["\[Beta]", "i"], "=",
  RowBox[{"(", RowBox[{
    RowBox[{"-", vsBoxed, " ", fOfRs}], ",", " ", "0", ",", " ", "0"}], ")"}],
  ",", "\[ThickSpace]\[ThickSpace]\[ThickSpace]",
  RowBox[{"\[Alpha]", "=", "1"}], ",", "\[ThickSpace]\[ThickSpace]",
  RowBox[{gammaIJ, "=", SubscriptBox["\[Delta]", RowBox[{"i", "j"}]]}]}];

g00Eq = RowBox[{SubscriptBox["g", "00"], "=",
  RowBox[{"-", SuperscriptBox["\[Alpha]", "2"], "+",
    RowBox[{SubscriptBox["\[Beta]", "i"], " ",
      SuperscriptBox["\[Beta]", "i"]}]}]}];

g0iEq = RowBox[{SubscriptBox["g", "0i"], "=", SubscriptBox["\[Beta]", "i"],
  "=", RowBox[{gammaIJ, " ", SuperscriptBox["\[Beta]", "j"]}]}];

gIjEq = RowBox[{SubscriptBox["g", "ij"], "=", gammaIJ}];

detAlcEq = RowBox[{RowBox[{"det", "(", ggMnLow, ")"}], "=",
  RowBox[{"-", SuperscriptBox["\[Alpha]", "2"], " ",
    RowBox[{"det", "(", gammaIJ, ")"}]}],
  "=", RowBox[{"-", "1"}]}];

(* ----- Section 3 extras (Christoffel/Riemann/Ricci/Einstein) ----- *)

christoffelEq = RowBox[{
  SubsuperscriptBox["\[CapitalGamma]", RowBox[{"\[Alpha]", "\[Beta]"}], "\[Mu]"],
  "=", RowBox[{FractionBox["1", "2"], " ",
    SuperscriptBox["g", RowBox[{"\[Mu]", "\[Nu]"}]], " ",
    RowBox[{"(",
      RowBox[{
        RowBox[{SubscriptBox["\[PartialD]", "\[Alpha]"], " ",
                SubscriptBox["g", RowBox[{"\[Beta]", "\[Nu]"}]]}],
        "+",
        RowBox[{SubscriptBox["\[PartialD]", "\[Beta]"], " ",
                SubscriptBox["g", RowBox[{"\[Alpha]", "\[Nu]"}]]}],
        "-",
        RowBox[{SubscriptBox["\[PartialD]", "\[Nu]"], " ",
                SubscriptBox["g", RowBox[{"\[Alpha]", "\[Beta]"}]]}]}],
      ")"}]}]}];

riemannEq = RowBox[{
  SubsuperscriptBox["R", RowBox[{"\[Sigma]", "\[Mu]", "\[Nu]"}], "\[Rho]"],
  "=",
  RowBox[{
    RowBox[{SubscriptBox["\[PartialD]", "\[Mu]"], " ",
      SubsuperscriptBox["\[CapitalGamma]", RowBox[{"\[Nu]", "\[Sigma]"}], "\[Rho]"]}],
    "-",
    RowBox[{SubscriptBox["\[PartialD]", "\[Nu]"], " ",
      SubsuperscriptBox["\[CapitalGamma]", RowBox[{"\[Mu]", "\[Sigma]"}], "\[Rho]"]}],
    "+",
    RowBox[{SubsuperscriptBox["\[CapitalGamma]", RowBox[{"\[Mu]", "\[Lambda]"}], "\[Rho]"],
      " ",
      SubsuperscriptBox["\[CapitalGamma]", RowBox[{"\[Nu]", "\[Sigma]"}], "\[Lambda]"]}],
    "-",
    RowBox[{SubsuperscriptBox["\[CapitalGamma]", RowBox[{"\[Nu]", "\[Lambda]"}], "\[Rho]"],
      " ",
      SubsuperscriptBox["\[CapitalGamma]", RowBox[{"\[Mu]", "\[Sigma]"}], "\[Lambda]"]}]}]}];

ricciContractionEq = RowBox[{RMnLow, "=",
  SubsuperscriptBox["R", RowBox[{"\[Mu]", "\[Rho]", "\[Nu]"}], "\[Rho]"],
  ",", "\[ThickSpace]\[ThickSpace]\[ThickSpace]",
  RowBox[{"R", "=",
    RowBox[{SuperscriptBox["g", RowBox[{"\[Mu]", "\[Nu]"}]], " ", RMnLow}]}]}];

einsteinDefEq = RowBox[{GMnLow, "=", RowBox[{RMnLow, "-",
  RowBox[{FractionBox["1", "2"], " ", ggMnLow, " ", "R"}]}]}];

(* ----- Section 4 extras (extrinsic curvature, expansion) ----- *)

extrinsicEq = RowBox[{KijLow, "=",
  RowBox[{FractionBox["1", RowBox[{"2", "\[Alpha]"}]],
    RowBox[{"(",
      RowBox[{
        RowBox[{SubscriptBox["D", "i"], " ", SubscriptBox["\[Beta]", "j"]}],
        "+",
        RowBox[{SubscriptBox["D", "j"], " ", SubscriptBox["\[Beta]", "i"]}],
        "-",
        RowBox[{SubscriptBox["\[PartialD]", "t"], " ", gammaIJ}]}],
      ")"}]}]}];

thetaTraceEq = RowBox[{"\[Theta]", "=",
  RowBox[{SuperscriptBox["\[Gamma]", RowBox[{"i", "j"}]], " ", KijLow}],
  "=",
  RowBox[{SubscriptBox["D", "i"], " ", SuperscriptBox["\[Beta]", "i"]}]}];

(* ----- Section 7 extras (geodesic) ----- *)

geodesicSystemEq = RowBox[{RowBox[{
  FractionBox[RowBox[{"d", SuperscriptBox["x", "\[Mu]"]}], "d\[Lambda]"],
  "=", SuperscriptBox["u", "\[Mu]"]}], ",", "\[ThickSpace]\[ThickSpace]\[ThickSpace]",
  RowBox[{
  FractionBox[RowBox[{"d", SuperscriptBox["u", "\[Mu]"]}], "d\[Lambda]"],
  "=", RowBox[{"-",
    SubsuperscriptBox["\[CapitalGamma]", RowBox[{"\[Alpha]", "\[Beta]"}], "\[Mu]"], " ",
    SuperscriptBox["u", "\[Alpha]"], " ",
    SuperscriptBox["u", "\[Beta]"]}]}]}];

normTimelikeEq = RowBox[{ggMnLow, " ",
  SuperscriptBox["u", "\[Mu]"], " ", SuperscriptBox["u", "\[Nu]"],
  "=", RowBox[{"-", "1"}]}];

normNullEq = RowBox[{ggMnLow, " ",
  SuperscriptBox["k", "\[Mu]"], " ", SuperscriptBox["k", "\[Nu]"],
  "=", "0"}];

(* ----- Section 8 extras (per-metric distinguishing math) ----- *)

natarioCondEq = RowBox[{"\[Del]", "\[CenterDot]", "\[Beta]", "=", "0"}];

natarioCurlEq = RowBox[{"\[Beta]", "=", RowBox[{"\[Del]", "\[Times]", "A"}]}];

vdbMetricEq = RowBox[{
  SuperscriptBox["ds", "2"], "=",
  RowBox[{"-", SuperscriptBox["dt", "2"]}], "+",
  SuperscriptBox[RowBox[{"B", "(", rsBoxed, ")"}], "2"], " ",
  RowBox[{"[", RowBox[{
    SuperscriptBox[RowBox[{"(", RowBox[{"dx", "-",
      RowBox[{vsBoxed, " ", fOfRs, " ", "dt"}]}], ")"}], "2"],
    "+", SuperscriptBox["dy", "2"], "+", SuperscriptBox["dz", "2"]}],
  "]"}]}];

vdbScalingEq = RowBox[{SubscriptBox["E", "VdB"], "\[Proportional]",
  SubscriptBox["R", "ext"], ",", "\[ThickSpace]\[ThickSpace]\[ThickSpace]",
  SubscriptBox["V", "interior"], "\[Proportional]",
  RowBox[{SuperscriptBox["B", "3"], " ",
    SuperscriptBox[SubscriptBox["R", "int"], "3"]}]}];

bobrickShellEq = RowBox[{
  RowBox[{"\[Chi]", "(", "r", ")"}], "=",
  RowBox[{
    FractionBox["1", "2"],
    RowBox[{"(", RowBox[{"1", "+",
      RowBox[{"tanh", "(",
        RowBox[{"\[Sigma]", "(", RowBox[{"r", "-",
          SubscriptBox["R", "in"]}], ")"}], ")"}]}], ")"}], " ",
    FractionBox["1", "2"],
    RowBox[{"(", RowBox[{"1", "-",
      RowBox[{"tanh", "(",
        RowBox[{"\[Sigma]", "(", RowBox[{"r", "-",
          SubscriptBox["R", "out"]}], ")"}], ")"}]}], ")"}]}]}];

bobrickSpeedEq = RowBox[{vsBoxed, "<", "1", "\[ThickSpace]\[ThickSpace]\[ThickSpace]",
  "\[Implies]", "\[ThickSpace]\[ThickSpace]",
  RowBox[{"WEC", " ", "satisfiable"}]}];

lentzAnsatzEq = RowBox[{
  RowBox[{"\[Phi]", "(", "r", ")"}], "=", RowBox[{"sech", "(",
    FractionBox["r", "R"], ")"}]}];

(* ----- Section 9 extras (energy conditions) ----- *)

secEq = RowBox[{
  RowBox[{"(", RowBox[{TMnLow, "-",
    RowBox[{FractionBox["1", "2"], " ", "T", " ", ggMnLow}]}], ")"}],
  " ", SuperscriptBox["u", "\[Mu]"], " ", SuperscriptBox["u", "\[Nu]"],
  "\[GreaterEqual]", "0"}];

decEq = RowBox[{
  "WEC", "+", "\[ThickSpace]",
  RowBox[{
    RowBox[{"-", SubsuperscriptBox["T", "\[Nu]", "\[Mu]"], " ",
      SuperscriptBox["u", "\[Nu]"]}],
    "\[ThickSpace]", "non-spacelike"}]}];

(* ---------- the cells --------------------------------------------- *)

cells = {

 Cell["Warp Drives in the Wolfram Language", "Title"],

 Cell["A symbolic-tensor playground for general-relativistic warp \
bubbles \[LongDash] from the Alcubierre metric to subluminal \
Bobrick\[Dash]Martire shells, with native Einstein tensors, \
geodesics, and animated spacetime distortions.", "Subtitle"],

 Cell[TextData[{"Marco Thiel, April 2026 \[LongDash] ",
                ButtonBox["github.com/mthiel74/WarpDrives",
                  BaseStyle -> "Hyperlink",
                  ButtonData -> {URL[
                    "https://github.com/mthiel74/WarpDrives"], None}]}],
      "Text", TextAlignment -> Center, FontSlant -> Italic],

 imgCell[loadAnim["anim_hero_3d.gif", 480, 28, 9],
         "Figure 0. The hero animation.  A stylised spacecraft sits \
inside an Alcubierre bubble travelling at v = c.  The cyan grid \
shows how the warp field lifts and bunches the local coordinate \
mesh; the translucent blue sphere is the bubble surface; the \
magenta ring at the equator is where the exotic matter (negative \
energy density) lives.  Engine wake, starfield, and orbiting camera \
are just for atmosphere \[LongDash] the geometry, the grid, the \
ring and the spacecraft pose are computed from the analytic \
Alcubierre metric every frame.  The full-size version of the same \
animation is in Section 5b below."],

 (* ----------- Abstract ---------- *)

 Cell["Abstract", "Section"],

 Cell[TextData[{
   "Warp drives are the most science-fiction-sounding object in \
classical general relativity.  In Alcubierre's 1994 construction, a \
spacecraft sits inside a flat \"bubble\" of spacetime that contracts \
the geometry in front of it and expands the geometry behind it.  \
Locally the ship never moves faster than light \[LongDash] but the \
bubble itself can carry the ship across arbitrary coordinate \
distances in arbitrarily short coordinate times.  The catch is the \
stress-energy: the metric ",
   StyleBox["requires", FontSlant -> Italic],
   " a region of negative energy density in the bubble wall.  This \
notebook ports the ",
   ButtonBox["WarpDrives Python package",
     BaseStyle -> "Hyperlink",
     ButtonData -> {URL[
       "https://github.com/mthiel74/WarpDrives"], None}],
   " into Wolfram Language packages and shows what symbolic-tensor \
machinery and ",
   StyleBox["DensityPlot", "Input"], " / ",
   StyleBox["NDSolve", "Input"],
   " buy you on this material: a working Einstein tensor in three \
lines, animated bubble fly-bys, geodesics that get swept up by the \
warp field, and side-by-side visualisations of the six metric \
families in the literature (Alcubierre 1994, Nat\[AAcute]rio 2002, \
Van Den Broeck 1999, White toroidal, Bobrick\[Dash]Martire 2021, \
Lentz 2021)."}],
      "Text"],

 Cell[TextData[{
   StyleBox["Status \[LongDash] hypothetical physics, not engineering.",
            FontWeight -> Bold],
   "  Every metric discussed below is a ",
   StyleBox["mathematical", FontSlant -> Italic],
   " solution of Einstein's field equations in the ",
   StyleBox["reverse", FontSlant -> Italic],
   " sense: one prescribes a desired geometry ",
   mathI[ggMnLow], " and reads off the stress-energy tensor ",
   mathI[TMnLow], " = ", mathI[GMnLow],
   "/(8\[Pi]G) that would be required to source it.  No known \
substance produces the resulting ",
   mathI[TMnLow],
   ".  In particular, all superluminal warp metrics treated here \
(Alcubierre, Nat\[AAcute]rio, Van Den Broeck, White, and the \
superluminal Lentz branch) violate the null energy condition \
(NEC), so they require ",
   StyleBox["exotic matter", FontWeight -> Bold],
   " \[LongDash] regions of negative energy density as measured by \
some observer.  Classical matter (dust, fluids, electromagnetic \
fields, scalar fields with canonical kinetic terms) cannot supply \
this; it is forbidden by every classical energy condition.  \
Quantum field theory does permit small, transient negative energy \
densities (the Casimir effect, squeezed vacuum states), but the \
Ford\[Dash]Roman quantum inequalities (1995\[Dash]1997) and \
Pfenning\[Dash]Ford's bubble-specific bound (1997) show that the \
negative-energy budget for an Alcubierre bubble of macroscopic \
size scales catastrophically: a bubble of radius ~100\[ThinSpace]m \
moving at ", mathI["v"], " = ", mathI["c"],
   " requires a wall thinner than the Planck length and a total \
negative mass-energy of order ",
   mathI[SuperscriptBox["10", "65"]],
   " grams \[LongDash] roughly ten billion times the mass of the \
observable universe, with the opposite sign.  Even granting the \
matter, superluminal bubbles develop horizons (Finazzi, Liberati \
& Barcel\[OAcute] 2009) that disconnect the ship from the front \
wall and exhibit semiclassical instabilities on timescales \
comparable to or shorter than the Planck time.  The genuinely \
subluminal proposal in this notebook (Bobrick\[Dash]Martire 2021) \
is the only one that can satisfy the weak and null energy \
conditions, but it still demands astrophysical energy densities; \
the Lentz 2021 hyper-fast soliton was originally advertised as \
positive-energy, but subsequent analyses suggest the construction \
still violates the classical energy conditions.  Treat what \
follows as ",
   StyleBox["differential geometry of conceivable spacetimes",
            FontSlant -> Italic],
   ", not a propulsion roadmap."}],
      "Text"],

 Cell["Getting started", "Section"],

 Cell[TextData[{"Every figure and animation below is already \
rendered, so the post reads top-to-bottom in the browser \[LongDash] \
the animated GIFs embedded in each Output cell play directly in the \
Wolfram Community viewer.  If you'd like to rerun or tweak any of \
them locally, the Input cell above each figure contains the exact \
call.  Everything needed sits in a single self-contained package, ",
   StyleBox["WarpBubbleSim.wl", "Input"],
   " (~30 KB, attached to this post).  Drop it next to the notebook, \
evaluate the cell below once, and every public symbol used later \
(",
   StyleBox["AlcubierreMetric", "Input"], ", ",
   StyleBox["EinsteinTensor", "Input"], ", ",
   StyleBox["CheckAllConditions", "Input"], ", ",
   StyleBox["IntegrateGeodesic", "Input"], ", \[Ellipsis]) becomes \
available:"}], "Text"],

 inputCell["SetDirectory[NotebookDirectory[]];
Get[\"WarpBubbleSim.wl\"];   (* one file: 65 public symbols covering
                                 shape functions, six warp metrics,
                                 Christoffel/Riemann/Einstein tensors,
                                 ADM tools, energy conditions,
                                 geodesics, plotting helpers. *)"],

 Cell[TextData[{"Source repository (Wolfram + Python ports side by \
side): ",
   ButtonBox["github.com/mthiel74/WarpDrives",
     BaseStyle -> "Hyperlink",
     ButtonData -> {URL[
       "https://github.com/mthiel74/WarpDrives"], None}], "."}],
      "Text"],

 (* ----------- Section 1: What is a warp drive? ---------- *)

 Cell["1. What is a warp drive?", "Section"],

 Cell[TextData[{
   "In special relativity nothing locally outpaces light, and that \
sets a hard ceiling on travel times: at half the speed of light, the \
nearest star is still nine years away.  General relativity is \
softer.  Once spacetime can curve, the question \"how fast did you \
travel?\" splits into two questions \[LongDash] how fast did you \
move with respect to the local geometry (still bounded by ",
   StyleBox["c", FontSlant -> Italic],
   "), and how fast did the geometry itself rearrange around you?  \
Cosmology answers the second question routinely: distant galaxies \
recede from us superluminally because the universe is expanding, \
even though no individual galaxy moves faster than light through its \
own neighbourhood."}],
      "Text"],

 Cell[TextData[{"Miguel Alcubierre asked in 1994 whether the same \
trick can be played locally and on demand.  The answer is yes: a \
sufficiently inventive metric can carry a flat-space \"bubble\" \
across the universe at any chosen ",
   mathI[vsBoxed], " by contracting space in front of the bubble and \
expanding space behind it.  The price tag, as we'll see in Section 9, \
is steep.  But the geometry itself is surprisingly clean and admits \
a famously short closed form:"}], "Text"],

 mathD[alcMetricEq],

 Cell[TextData[{"where ", mathI[vsBoxed],
   " is the bubble's coordinate velocity, ",
   mathI[xsBoxed], " its centre, with trajectory"}], "Text"],

 mathD[xsTrajectoryEq],

 Cell[TextData[{"and ", mathI[rsBoxed],
   " is the Euclidean distance from the bubble centre on the spatial slice"}],
   "Text"],

 mathD[rsDefEq],

 Cell[TextData[{"so a static observer at fixed ",
   mathI[RowBox[{"(", "x", ",", "y", ",", "z", ")"}]],
   " sees ", mathI[rsBoxed], " shrink linearly to zero as the bubble \
flies past, then grow back.  The shape function ", mathI[fOfRs],
   " is what gives the bubble a finite extent; mathematically it can \
be any smooth scalar with the asymptotic behaviour"}], "Text"],

 mathD[fLimitEq],

 Cell[TextData[{"Inside the bubble (",
   mathI[RowBox[{rsBoxed, "\[LessEqual]", "R"}]], ", ",
   mathI[RowBox[{"f", "\[TildeTilde]", "1"}]],
   ") the line element collapses to ",
   mathI[RowBox[{
     SuperscriptBox["ds", "2"], "=",
     RowBox[{"-", SuperscriptBox["dt", "2"], "+",
       SuperscriptBox[
         RowBox[{"(", RowBox[{"dx", "-", RowBox[{vsBoxed, " ", "dt"}]}],
                  ")"}], "2"],
       "+", SuperscriptBox["dy", "2"], "+", SuperscriptBox["dz", "2"]}]}]],
   ", which is just Minkowski space written in coordinates that are \
boosted at velocity ", mathI[vsBoxed],
   " \[LongDash] no curvature, no acceleration, the spacecraft sails \
along a straight worldline.  Outside the bubble (",
   mathI[RowBox[{rsBoxed, ">>", "R"}]],
   ") ", mathI[fOfRs], " is essentially zero and the metric is \
again Minkowski.  All the curvature and exotic-matter cost is \
concentrated in the ",
   StyleBox["wall region", FontSlant -> Italic],
   " where ", mathI[fOfRs], " transitions from 1 to 0."}], "Text"],

 Cell[TextData[{"Alcubierre's original shape function is the Tanh \
sandwich"}], "Text"],

 mathD[shapeFnEq],

 Cell[TextData[{"with two free parameters: ", mathI["R"],
   " sets the bubble radius, ", mathI[sigmaSym],
   " controls the wall thickness (",
   mathI[RowBox[{"\[Sigma]", "\[Rule]", "\[Infinity]"}]],
   " gives a hard step, ", mathI[RowBox[{"\[Sigma]", "\[Rule]", "0"}]],
   " smears the wall over all of space).  This is the canonical \
choice but other smooth profiles work too \[LongDash] a Gaussian, a \
compact-support polynomial, a smoothstep \[LongDash] each with a \
different trade-off between wall sharpness and energy density:"}],
      "Text"],

 inputCell[
"Plot[{TanhShape[r, 1, 8],
      GaussianShape[r, 1, 0.4],
      CompactPolynomialShape[r, 1, 0.4],
      SmoothStepShape[r, 1, 0.3]}, {r, 0, 2.5},
  PlotLegends -> {\"Tanh\", \"Gaussian\", \"Polynomial\", \"Smoothstep\"},
  Frame -> True, FrameLabel -> {\"r / R\", \"f(r)\"}]"],

 imgCell[loadPng["shape_functions.png"]],

 Cell[TextData[{"Each profile satisfies the same boundary \
conditions \[LongDash] ",
   mathI[RowBox[{"f", "(", "0", ")", "=", "1"}]], ", ",
   mathI[RowBox[{"f", "(", "\[Infinity]", ")", "=", "0"}]],
   " \[LongDash] but they differ in how sharp the wall is and \
whether the support is compact.  Sharper walls localise the exotic \
matter into a thinner shell at the cost of larger derivatives, \
which means more curvature, which means more energy.  This is the \
first hint that the energy bill of a warp drive is essentially set \
by the ",
   StyleBox["geometry of the wall", FontSlant -> Italic],
   ", not by the bubble's velocity."}], "Text"],

 (* ----------- Section 2: The metric in ADM form ---------- *)

 Cell["2. The metric in ADM form", "Section"],

 Cell[TextData[{"Every warp drive in the literature is most natural \
in the 3+1 ADM formulation.  Foliate spacetime into spatial slices \
labelled by ", mathI["t"], "; on each slice keep the spatial metric ",
   mathI[gammaIJ],
   "; relate proper time on neighbouring slices by the lapse ",
   mathI[alphaSym], "; relate spatial coordinates between slices by \
the shift vector ", mathI[betaUp],
   ".  The line element splits into a temporal-distance piece and a \
spatial-displacement piece:"}], "Text"],

 mathD[admEq],

 Cell[TextData[{"Reading off matrix entries gives the standard 3+1 \
decomposition of the 4-metric:"}], "Text"],

 mathD[g00Eq], mathD[g0iEq], mathD[gIjEq],

 Cell[TextData[{"with summation convention on the spatial indices ",
   mathI[RowBox[{"i", ",", "j", "\[Element]", RowBox[{"{", "1", ",", "2",
     ",", "3", "}"}]}]],
   " and the spatial-metric raise/lower ",
   mathI[RowBox[{SubscriptBox["\[Beta]", "i"], "=",
     RowBox[{gammaIJ, " ", SuperscriptBox["\[Beta]", "j"]}]}]],
   ".  The inverse 4-metric has a similarly clean form: ",
   mathI[RowBox[{SuperscriptBox["g", "00"], "=",
     RowBox[{"-", SuperscriptBox["\[Alpha]", RowBox[{"-", "2"}]]}]}]],
   ", ", mathI[RowBox[{SuperscriptBox["g", "0i"], "=",
     RowBox[{SuperscriptBox["\[Beta]", "i"],
       SuperscriptBox["\[Alpha]", RowBox[{"-", "2"}]]}]}]],
   ", ", mathI[RowBox[{SuperscriptBox["g", "ij"], "=",
     RowBox[{SuperscriptBox["\[Gamma]", RowBox[{"i", "j"}]], "-",
       SuperscriptBox["\[Beta]", "i"], " ", SuperscriptBox["\[Beta]", "j"],
       " ", SuperscriptBox["\[Alpha]", RowBox[{"-", "2"}]]}]}]],
   ", which is exactly what ", StyleBox["ADMInverse[\[Alpha], \
\[Beta], \[Gamma]]", "Input"], " in the package returns."}], "Text"],

 Cell[TextData[{"For Alcubierre's drive the ADM ingredients are \
spectacularly simple:"}], "Text"],

 mathD[alcShiftEq],

 Cell[TextData[{"All the warp \"weirdness\" lives in that single ",
   mathI[betaUp],
   " term \[LongDash] a unit-magnitude vector field along the \
direction of motion, modulated by the shape function so it switches \
on inside the bubble and off outside.  In Wolfram, the helper ",
   StyleBox["AlcubierreMetric", "Input"],
   " builds the full 4-metric symbolically:"}], "Text"],

 inputCell[
"g = AlcubierreMetric[{1, 1, 8}, {t, x, y, z}];   (* {v0, R, \[Sigma]} *)
MatrixForm @ g"],

 Cell[TextData[{"With ", mathI[RowBox[{"\[Alpha]", "=", "1"}]],
   " and ", mathI[RowBox[{"det", "(", gammaIJ, ")", "=", "1"}]],
   " the formula above gives ", mathI[RowBox[{"det", "(", ggMnLow, ")",
     "=", RowBox[{"-", "1"}]}]],
   " everywhere, and Wolfram simplifies ",
   StyleBox["Det[g]", "Input"],
   " back to that exact value:"}], "Text"],

 inputCell[
"Simplify @ Det @ AlcubierreMetric[{v, R, s}, {t, x, y, z}]
(* -> -1 *)"],

 (* ----------- Section 3: Symbolic Einstein tensor ---------- *)

 Cell["3. Einstein tensor in three lines", "Section"],

 Cell[TextData[{"This is where Wolfram earns its keep.  In the \
Python port we built finite-difference Christoffel symbols, then \
finite-difference Riemann tensors, then finite-difference Ricci, all \
~500 lines.  In Wolfram the equivalent is a literal transliteration \
of the textbook formula because ",
   StyleBox["D", "Input"],
   " differentiates symbolically.  Recall the Christoffel symbols of \
the second kind:"}], "Text"],

 mathD[christoffelEq],

 Cell[TextData[{"In Wolfram this is one Table loop:"}], "Text"],

 inputCell[
"ChristoffelSymbols[g_, {t_, x_, y_, z_}] := Module[
  {ginv = Inverse[g],
   dg = Table[D[g, c], {c, {t, x, y, z}}]},
  Table[
    (1/2) Sum[ginv[[m, n]] (dg[[a, b, n]] + dg[[b, a, n]] - dg[[n, a, b]]),
              {n, 4}],
    {m, 4}, {a, 4}, {b, 4}]];"],

 Cell[TextData[{"The Riemann tensor follows from the standard \
non-commutativity of covariant derivatives,"}], "Text"],

 mathD[riemannEq],

 Cell[TextData[{"and Ricci is a single contraction, the Ricci scalar \
a second one:"}], "Text"],

 mathD[ricciContractionEq],

 Cell[TextData[{"with ", mathI[SubscriptBox["\[PartialD]", "\[Mu]"]],
   " denoting partial differentiation with respect to ",
   mathI[SuperscriptBox["x", "\[Mu]"]],
   " and the usual Einstein summation convention.  The Einstein \
tensor is then a one-line definition:"}], "Text"],

 mathD[einsteinDefEq],

 Cell[TextData[{"Through the field equation"}], "Text"],

 mathD[einsteinEq],

 Cell[TextData[{"with ", mathI[RowBox[{"G", "=", "c", "=", "1"}]],
   ", we read off the matter content the metric implies:"}], "Text"],

 mathD[stressEq],

 Cell[TextData[{"In Wolfram that's a one-liner.  At a point on the \
bubble wall:"}], "Text"],

 inputCell[
"GAt = EinsteinTensor[g, {t, x, y, z}] /.
       {t -> 0, x -> 1.0, y -> 0.2, z -> 0};
N @ Chop @ GAt // MatrixForm
(*  the (0,0) entry is 8\[Pi]\[Rho]_Eulerian, off-diagonal entries are
    energy fluxes, ij-block is anisotropic stress  *)"],

 Cell[TextData[{"For Alcubierre the energy density seen by a static \
observer admits a famous closed form, derived by Alcubierre in the \
original paper:"}], "Text"],

 mathD[rhoAnalytic],

 Cell[TextData[{"Plotted in the ",
   mathI[RowBox[{"z", "=", "0"}]],
   " plane it's strikingly anisotropic \[LongDash] zero on the \
direction-of-motion axis, peaking in two crescent-shaped regions in \
the bubble wall.  Note the colour scale: the values are ",
   StyleBox["negative everywhere", FontSlant -> Italic],
   " they are non-zero.  This is the punchline of the original \
1994 paper:"}], "Text"],

 inputCell[
"DensityPlot[
  AlcubierreAnalyticEnergyDensity[{1, 1, 8}, {0, x, y, 0}],
  {x, -2.5, 2.5}, {y, -2.5, 2.5},
  PlotPoints -> 220, ColorFunction -> \"TemperatureMap\",
  PlotLegends -> Automatic,
  Epilog -> {Black, Thick, Circle[{0, 0}, 1]}]"],

 imgCell[loadPng["alcubierre_density.png"]],

 Cell[TextData[{"And the same energy density rendered as a 3D \
surface brings out the wall structure:"}], "Text"],

 imgCell[loadPng["energy_density_3d.png"],
   "Figure 1. Eulerian energy density of an Alcubierre bubble (v=c, \
R=1, \[Sigma]=8).  Two depressions in the bubble wall \[LongDash] one \
in front of the ship, one behind \[LongDash] is where the exotic \
matter (\[Rho] < 0) must live."],

 (* ----------- Section 4: The expansion scalar ---------- *)

 Cell["4. Where space contracts, where space expands", "Section"],

 Cell[TextData[{"What does \"contraction in front, expansion behind\" \
mean operationally?  The ADM expansion scalar of the Eulerian \
observers measures it directly.  An Eulerian observer is one that \
moves with the spatial slices \[LongDash] their 4-velocity is the \
unit normal ",
   mathI[RowBox[{SuperscriptBox["n", "\[Mu]"], "=",
     RowBox[{SuperscriptBox["\[Alpha]", RowBox[{"-", "1"}]],
       RowBox[{"(", RowBox[{"1", ",", " ",
         RowBox[{"-", SuperscriptBox["\[Beta]", "i"]}]}], ")"}]}]}]],
   ".  The extrinsic curvature of the slice measures how it bends \
inside spacetime as time flows:"}], "Text"],

 mathD[extrinsicEq],

 Cell[TextData[{"with ", mathI[SubscriptBox["D", "i"]],
   " the spatial covariant derivative on the slice (compatible with ",
   mathI[gammaIJ],
   ").  Its trace is the volume expansion rate seen by Eulerian \
observers:"}], "Text"],

 mathD[thetaTraceEq],

 Cell[TextData[{"For Alcubierre's metric ",
   mathI[RowBox[{"\[Alpha]", "=", "1"}]],
   " and ", mathI[RowBox[{gammaIJ, "=", SubscriptBox["\[Delta]",
     RowBox[{"i", "j"}]]}]],
   ", so ", mathI[RowBox[{SubscriptBox["D", "i"], " ",
     SuperscriptBox["\[Beta]", "i"], "=",
     RowBox[{SubscriptBox["\[PartialD]", "i"], " ",
       SuperscriptBox["\[Beta]", "i"]}]}]],
   " is just the ordinary divergence.  Plugging in ",
   mathI[RowBox[{SuperscriptBox["\[Beta]", "x"], "=",
     RowBox[{"-", vsBoxed, " ", fOfRs}]}]],
   " and using the chain rule one gets a strikingly clean closed \
form:"}], "Text"],

 mathD[thetaEq],

 Cell[TextData[{"For Alcubierre's bubble it's antisymmetric in ",
   mathI[RowBox[{"x", "-", xsBoxed}]],
   ": positive (volume expansion) behind the bubble, negative \
(volume contraction) in front, zero on the y-z plane through the \
centre.  Plotted on a slice at ", mathI[RowBox[{"z", "=", "0"}]],
   " using a divergent colourmap:"}], "Text"],

 inputCell[
"DensityPlot[
  AlcubierreExpansionScalar[{1, 1, 8}, {0, x, y, 0}],
  {x, -2.5, 2.5}, {y, -2.5, 2.5},
  PlotPoints -> 220, ColorFunction -> \"RedBlueTones\",
  PlotLegends -> Automatic,
  Epilog -> {Black, Thick, Circle[{0, 0}, 1]}]"],

 imgCell[loadPng["expansion_scalar.png"],
   "Figure 2. Expansion scalar \[Theta] of Eulerian observers around \
an Alcubierre bubble.  Blue = space contracting (front of bubble), \
red = space expanding (back of bubble).  The y-z plane through the \
centre is exactly neutral."],

 Cell[TextData[{"So if you stand inside the bubble and look out the \
front porthole, distant galaxies appear to be rushing towards you; \
look out the back, they're racing away.  The bubble itself \
\"surfs\" forward on its own private patch of contracting space.  \
This is the qualitative picture you'll have heard described in \
popular accounts \[LongDash] and the picture above is its actual \
mathematical content."}], "Text"],

 (* ----------- Section 5: Bubble flying through space ---------- *)

 Cell["5. The bubble in motion", "Section"],

 Cell[TextData[{"All of the above is for a bubble centred at the \
origin.  Plug in time and the centre advects at uniform velocity ",
   mathI[vsBoxed],
   ".  Animated, this gives the canonical \"warp drive\" image: a \
moving capsule of warped geometry, exotic-matter wall and all:"}],
      "Text"],

 inputCell[
"frames = Table[
  Show[
    DensityPlot[
      AlcubierreAnalyticEnergyDensity[{1, 1, 8}, {t, x, y, 0}],
      {x, -3.5, 5.5}, {y, -2, 2}, PlotPoints -> 130,
      ColorFunction -> \"TemperatureMap\", Frame -> True],
    Graphics[{Black, Thick, Circle[{1.0 t, 0}, 1.0]}]],
  {t, 0, 4, 4/29.}];
ListAnimate[frames]"],

 imgCell[loadAnim["anim_bubble_flying.gif", 600, 30, 6],
   "Figure 3. The same Alcubierre bubble flying across the frame \
at v = c.  The colourmap shows Eulerian energy density; the black \
circle is the nominal bubble radius R = 1.  Notice that the bubble \
geometry is rigidly carried along \[LongDash] no shockwaves, no \
wakes \[LongDash] precisely because the construction is engineered \
to be flat inside and outside."],

 (* ----------- Section 5b: The full hero animation ---------- *)

 Cell["5b. Putting a spacecraft inside the bubble", "Section"],

 Cell[TextData[{"None of the geometry above changes if we put a \
\"ship\" inside the bubble \[LongDash] in Alcubierre's metric the \
interior is exactly flat Minkowski, so the spacecraft is in inertial \
free-fall and feels no acceleration even as the bubble accelerates \
across the galaxy.  This makes for a striking visualisation, and \
because the package builds the metric symbolically we can render a \
fully self-consistent scene: the warp surface (a height field built \
from the shape function and the shift component), the translucent \
bubble, the magenta exotic-matter ring at the equator, the \
spacecraft mesh, an engine wake, and a star field.  The camera \
orbits the bubble while travelling with it, so the spacecraft stays \
centred while the stars stream past:"}], "Text"],

 Cell[TextData[{"The spacecraft itself is a small ",
   StyleBox["GraphicsComplex", "Input"],
   " with 17 vertices and 30 triangular faces, plus two flat-quad \
nacelles drawn in glowing blue.  The mesh is a literal port of the \
Python ", StyleBox["create_spacecraft()", "Input"],
   " in ", StyleBox["generate_hero_animation.py", "Input"], ":"}],
      "Text"],

 inputCell[
"hullLength = 0.6; hullWidth = 0.15; hullHeight = 0.08;
shipVerts  = (* nose, four front, four mid (widest), four rear,
                four engine \[LongDash] 17 vertices total *)
  { {hullLength 0.6, 0, 0},
    {hullLength 0.2,  hullWidth 0.3,  hullHeight 0.5},
    {hullLength 0.2, -hullWidth 0.3,  hullHeight 0.5},
    {hullLength 0.2, -hullWidth 0.3, -hullHeight 0.5},
    {hullLength 0.2,  hullWidth 0.3, -hullHeight 0.5},
    {0,  hullWidth 0.5,  hullHeight 0.7},
    {0, -hullWidth 0.5,  hullHeight 0.7},
    {0, -hullWidth 0.5, -hullHeight 0.5},
    {0,  hullWidth 0.5, -hullHeight 0.5},
    (* ... rear + engine vertices ... *)
    {-hullLength 0.5,  hullWidth 0.25, -hullHeight 0.2}};
shipFaces = {{1,2,3},{1,3,4},{1,4,5},{1,5,2}, (* nose cone *)
             (* ... 26 more triangles ... *) };
spacecraft3D[{xc_, yc_, zc_}, scale_] :=
  {RGBColor[0.78, 0.78, 0.82], Specularity[White, 30],
   GraphicsComplex[scale # + {xc, yc, zc} & /@ shipVerts,
                   Polygon[shipFaces]],
   Glow[RGBColor[0.25, 0.5, 1]], RGBColor[0.25, 0.5, 1],
   (* nacelles ... *),
   Glow[Cyan], Cyan,
   Sphere[{xc - 0.3 scale, yc, zc}, 0.05 scale]};"],

 Cell[TextData[{"The full mesh definition lives in ",
   StyleBox["build_hero_assets.wls", "Input"],
   ", which also produces the GIF.  Built into a 60-frame scene with \
an orbiting camera and a starfield, the result is the cover \
animation reproduced below at full size:"}], "Text"],

 imgCell[loadAnim["anim_hero_3d.gif", 640, 40, 10],
   "Figure 4. The spacecraft hero animation.  The cyan grid is a \
height-field visualisation of \[Minus]f(r) (geometric distortion of \
the warp surface) plus the shift-component cosmetic offset \
\[Beta]^x.  The blue translucent sphere is the bubble surface (R = \
1.5).  The magenta annular disc just below the equator is the \
exotic-matter ring \[LongDash] coloured red where the analytic \
Eulerian \[Rho] is most negative.  Notice that the cyan trail \
behind the ship is purely cosmetic: in Alcubierre's metric the \
bubble leaves no wake, since the geometry behind it is flat."],

 Cell[TextData[{"The 3D scene is striking but it makes the \
geometry hard to read.  The complementary top-down view drops the \
height-field and lays everything flat: the energy density as the \
backdrop (red where exotic matter must live), the cyan grid still \
distorted by the shift field, and a triangular silhouette of the \
ship with two glowing nacelle dots.  Annotations point at where \
space contracts and where it expands:"}], "Text"],

 imgCell[loadAnim["anim_hero_topdown.gif", 680, 36, 10],
   "Figure 5. Top-down view of the same scene.  Same metric, same \
bubble velocity, same spacecraft, but no z-axis cosmetics so the \
geometry is unobstructed.  The red ring is the actual analytic \
\[Rho] (more red = more exotic matter required); the cyan grid is \
the coordinate mesh advected by the shift field; the dashed white \
circle is the nominal bubble radius."],

 Cell[TextData[{"Both animations are rendered entirely from the \
analytic Alcubierre formulas \[LongDash] no GR solver, no NDSolve, \
just the closed forms for the shape function ", mathI[fOfRs],
   ", the shift ",
   mathI[RowBox[{SuperscriptBox["\[Beta]", "x"], "=", "-",
     vsBoxed, " ", fOfRs}]],
   ", and the Eulerian energy density.  Both run in seconds.  The \
spacecraft mesh and animation logic add fewer than 200 lines of \
Wolfram on top of the existing package."}], "Text"],

 (* ----------- Section 6: Coordinate-grid distortion ---------- *)

 Cell["6. The grid being warped", "Section"],

 Cell[TextData[{"Another way to see what the bubble is doing to the \
geometry is to drop a regular Cartesian grid into the spacetime and \
let the shift field advect every grid intersection.  As the bubble \
sweeps from left to right across the frame the grid lines \
locally bunch up in front of it and stretch out behind it.  The \
animation below uses a 14\[Times]14 grid, the analytic Alcubierre \
shift field with v=c and R=1, and a simple Euler advection step:"}],
   "Text"],

 inputCell[
"alcShift[t_, x_, y_] := -1.0 TanhShape[Sqrt[(x - 1.0 t)^2 + y^2], 1, 8];

advect[{x0_, y0_}, t_, nstep_] := Module[{x = x0, y = y0, dt = t/nstep},
  Do[{x, y} = {x + dt alcShift[k dt, x, y], y}, {k, 0, nstep - 1}];
  {x, y}];

frame[t_] := Show[
  DensityPlot[Abs[alcShift[t, x, y]], {x, -3, 3}, {y, -2.5, 2.5},
    ColorFunction -> \"SunsetColors\", Frame -> True],
  Graphics[{Thickness[0.0015], Black,
    Table[Line[Table[advect[{x, yi}, t, 14], {x, -3, 3, 6/60}]],
          {yi, -3, 3, 6/14}],
    Table[Line[Table[advect[{xi, y}, t, 14], {y, -3, 3, 6/60}]],
          {xi, -3, 3, 6/14}]}]];
ListAnimate[Table[frame[t], {t, 0, 2, 2/29.}]]"],

 imgCell[loadAnim["anim_grid_distortion.gif", 540, 30, 6],
   "Figure 6. A regular Cartesian grid distorted by the Alcubierre \
shift field as the bubble sweeps from left to right.  The colour \
shows the shift magnitude.  Inside the bubble all four points of any \
small grid cell move together \[LongDash] no internal stretching, \
no tidal forces.  All the deformation is concentrated in the wall."],

 (* ----------- Section 7: Geodesics ---------- *)

 Cell["7. Geodesics: getting swept up by a warp bubble", "Section"],

 Cell[TextData[{"Drop a population of test particles into the \
spacetime and let them follow geodesics.  Free-fall trajectories ",
   mathI[RowBox[{SuperscriptBox["x", "\[Mu]"], "(", "\[Lambda]", ")"}]],
   " obey the geodesic equation"}], "Text"],

 mathD[geodesicEq],

 Cell[TextData[{"which we rewrite as a first-order ODE system for \
position and 4-velocity ",
   mathI[RowBox[{SuperscriptBox["u", "\[Mu]"], "=",
     RowBox[{"d", " ", SuperscriptBox["x", "\[Mu]"], "/", "d\[Lambda]"}]}]],
   ":"}], "Text"],

 mathD[geodesicSystemEq],

 Cell[TextData[{"For massive (timelike) particles the affine \
parameter ", mathI["\[Lambda]"], " can be chosen to be the proper \
time, and the velocity is normalised so that"}], "Text"],

 mathD[normTimelikeEq],

 Cell[TextData[{"For light rays (null geodesics) the normalisation \
becomes ", mathI[normNullEq],
   " instead, and ", mathI["\[Lambda]"],
   " is no longer proper time (which is zero for a photon) but any \
affine parameter along the ray.  The Christoffel coefficients ",
   mathI[SubsuperscriptBox["\[CapitalGamma]", RowBox[{"\[Alpha]",
     "\[Beta]"}], "\[Mu]"]],
   " in the right-hand side are the same expressions Wolfram \
generated symbolically in Section 3 \[LongDash] we just substitute \
the current ",
   mathI[SuperscriptBox["x", "\[Mu]"]], " values into them and hand \
the result to ",
   StyleBox["NDSolve", "Input"], "."}], "Text"],

 Cell[TextData[{"In ADM language an Eulerian observer (the analogue \
of a stationary one in flat space) has spatial 3-velocity ",
   mathI[RowBox[{"-", betaUp}]],
   ", so as the bubble approaches a stationary particle from the \
left, the particle is grabbed by the shift field and accelerated \
forward; as the bubble passes, the particle decelerates back to \
rest.  Particles further from the axis feel proportionally less of \
the bubble's grip:"}], "Text"],

 inputCell[
"alcShift[t_, x_, y_] := -1.0 TanhShape[Sqrt[(x - 1.0 t)^2 + y^2], 1, 8];

(* Eulerian advection: dx/dt = -beta^x = +alcShift *)
sweep[{x0_, y0_}, ts_] := FoldList[
  {#1[[1]] + (#2 - First@ts) alcShift[#2, #1[[1]], y0], #1[[2]]} &,
  {x0, y0}, Rest[ts]];

ts = Range[0., 4., 0.05];
trajs = Table[sweep[{-3., y0}, ts], {y0, Range[-1.4, 1.4, 0.35]}];
ListAnimate @ Table[
  Show[DensityPlot[Abs[alcShift[ts[[k]], x, y]], {x, -3, 5.5},
                   {y, -2, 2}, PlotPoints -> 90,
                   ColorFunction -> \"BlueGreenYellow\"],
       Graphics[{Black, Thick, Line /@ (Take[#, k] & /@ trajs),
         Red, PointSize[0.022], Point /@ (#[[k]] & /@ trajs)}]],
  {k, 1, Length[ts], 2}]"],

 imgCell[loadAnim["anim_geodesic_bundle.gif", 600, 28, 7],
   "Figure 7. Eight initially stationary test particles caught up by \
an oncoming Alcubierre bubble.  Particles near the axis are dragged \
fastest; off-axis particles barely move.  Black tails are the \
trajectories so far; red dots are the current positions.  The \
colourmap shows the shift magnitude, i.e. the local speed at which \
the bubble \"grabs\" passing fluid."],

 Cell[TextData[{"For a real geodesic integration (full 4-velocity, \
Christoffel-symbol-driven) the package ships ",
   StyleBox["IntegrateGeodesic", "Input"],
   " on top of ", StyleBox["NDSolve", "Input"],
   ".  The animation above uses the Eulerian-observer shortcut, \
which is exact for static observers in an ADM spacetime with ",
   mathI[RowBox[{alphaSym, "=", "1"}]], "."}], "Text"],

 (* ----------- Section 8: All six metrics ---------- *)

 Cell["8. Six warp drives in one figure", "Section"],

 Cell[TextData[{"Alcubierre's drive is the simplest member of a \
larger family of geometries.  Six are currently of interest in the \
literature, each addressing a specific physical objection to its \
predecessor:\n",
   "\[Bullet] ",
   StyleBox["Alcubierre 1994", FontWeight -> Bold],
   " \[LongDash] the original.  Negative energy required.\n",
   "\[Bullet] ",
   StyleBox["Nat\[AAcute]rio 2002", FontWeight -> Bold],
   " \[LongDash] same speed, but constructed so the shift vector is \
divergence-free.  No volume expansion or contraction \[LongDash] \
the bubble \"slides\" through space rather than \"pushing\" it.\n",
   "\[Bullet] ",
   StyleBox["Van Den Broeck 1999", FontWeight -> Bold],
   " \[LongDash] adds a \"pocket\" interior so the external bubble \
radius can be microscopic while the inside stays habitable.  Energy \
scales with the (small) external radius.\n",
   "\[Bullet] ",
   StyleBox["White toroidal", FontWeight -> Bold],
   " \[LongDash] heuristic NASA construction with the exotic matter \
distributed on a torus rather than a sphere.\n",
   "\[Bullet] ",
   StyleBox["Bobrick\[Dash]Martire 2021", FontWeight -> Bold],
   " \[LongDash] the breakthrough: subluminal warp drives can be \
sourced by ",
   StyleBox["positive", FontSlant -> Italic],
   " energy.  The trade-off is the speed limit ",
   mathI[RowBox[{vsBoxed, "<", "c"}]], ".\n",
   "\[Bullet] ",
   StyleBox["Lentz 2021", FontWeight -> Bold],
   " \[LongDash] soliton drive sourced by Einstein\[Dash]Maxwell\
\[Dash]plasma fields, claimed positive-energy.  Subsequent analysis \
suggests the original construction needed exotic matter after all, \
but the soliton family is still actively studied."}], "Text"],

 Cell[TextData[{"The six metric families are mathematically distinct \
even though they share the same broad qualitative picture.  In ADM \
form, each one differs only in the choice of lapse, shift, and \
spatial metric:"}], "Text"],

 Cell[TextData[{"\[Bullet] ",
   StyleBox["Alcubierre", FontWeight -> Bold], " takes ",
   mathI[RowBox[{"\[Alpha]", "=", "1"}]], ", ",
   mathI[RowBox[{gammaIJ, "=", SubscriptBox["\[Delta]",
     RowBox[{"i", "j"}]]}]], ", and ",
   mathI[RowBox[{betaUp, "=",
     RowBox[{"(", RowBox[{"-", vsBoxed, " ", fOfRs}], ",", " ", "0",
       ",", " ", "0", ")"}]}]],
   ", as we've seen.  The shift has non-zero divergence \
(", mathI[RowBox[{"\[Del]", "\[CenterDot]", "\[Beta]", "\[NotEqual]", "0"}]],
   ") so spacetime expands and contracts."}], "Text"],

 Cell[TextData[{"\[Bullet] ",
   StyleBox["Nat\[AAcute]rio", FontWeight -> Bold],
   " keeps the same flat ", mathI[gammaIJ], " and unit lapse but \
demands"}], "Text"],

 mathD[natarioCondEq],

 Cell[TextData[{"so the Eulerian observers see ",
   mathI[RowBox[{"\[Theta]", "=", "0"}]],
   " everywhere \[LongDash] no expansion, no contraction, just \
shear.  This is enforced by writing the shift as the curl of a \
vector potential,"}], "Text"],

 mathD[natarioCurlEq],

 Cell[TextData[{"with ", mathI["A"], " chosen so that the bubble \
moves at the desired velocity.  The bubble in this picture \"slides \
through\" space rather than \"pushing\" space."}], "Text"],

 Cell[TextData[{"\[Bullet] ",
   StyleBox["Van Den Broeck", FontWeight -> Bold],
   " keeps Alcubierre's shift but multiplies the spatial metric by a \
position-dependent conformal factor:"}], "Text"],

 mathD[vdbMetricEq],

 Cell[TextData[{"with ", mathI[RowBox[{"B", "(", rsBoxed, ")"}]],
   " chosen to be ",
   mathI[RowBox[{"B", "\[GreaterTilde]", " ", SubscriptBox["B", "int"]}]],
   " inside an inner radius ", mathI[SubscriptBox["R", "int"]],
   " and ", mathI[RowBox[{"B", "\[TildeTilde]", "1"}]],
   " outside an outer radius ", mathI[SubscriptBox["R", "ext"]],
   ".  This gives the bubble a microscopic external footprint while \
preserving a macroscopic interior \[LongDash] and crucially the \
energy scales with the small ", mathI[SubscriptBox["R", "ext"]],
   " not the large ", mathI[SubscriptBox["R", "int"]], ":"}], "Text"],

 mathD[vdbScalingEq],

 Cell[TextData[{"\[Bullet] ",
   StyleBox["White toroidal", FontWeight -> Bold],
   " replaces the spherical distance ", mathI[rsBoxed],
   " with the Euclidean distance to the surface of a torus,"}],
   "Text"],

 mathD[RowBox[{"d", "(", RowBox[{"x", ",", "y", ",", "z"}], ")", "=",
   RowBox[{
     SqrtBox[
       RowBox[{
         SuperscriptBox[RowBox[{"(",
           RowBox[{SqrtBox[RowBox[{SuperscriptBox["y", "2"], "+",
             SuperscriptBox["z", "2"]}]],
             "-", SubscriptBox["R", "major"]}], ")"}], "2"],
         "+",
         SuperscriptBox[RowBox[{"(", RowBox[{"x", "-", xsBoxed}], ")"}],
                        "2"]}]],
     "-", SubscriptBox["R", "minor"]}]}]],

 Cell[TextData[{"so the exotic-matter wall lies on a tube of major \
radius ", mathI[SubscriptBox["R", "major"]], " and minor radius ",
   mathI[SubscriptBox["R", "minor"]],
   ", concentrating the negative energy into a doughnut rather than \
a sphere."}], "Text"],

 Cell[TextData[{"\[Bullet] ",
   StyleBox["Bobrick\[Dash]Martire", FontWeight -> Bold],
   " is the most physically interesting: localise the metric \
deviation in a thin spherical shell ",
   mathI[RowBox[{SubscriptBox["R", "in"], "<", "r", "<",
     SubscriptBox["R", "out"]}]], " using"}], "Text"],

 mathD[bobrickShellEq],

 Cell[TextData[{"and pick the shell-region lapse, shift, and \
spatial-metric perturbations to satisfy the energy conditions.  \
The key result is:"}], "Text"],

 mathD[bobrickSpeedEq],

 Cell[TextData[{"i.e. once the bubble is forced to move slower than \
light, ordinary positive-energy matter is enough to source it."}],
      "Text"],

 Cell[TextData[{"\[Bullet] ",
   StyleBox["Lentz", FontWeight -> Bold],
   " uses a soliton ansatz with hyperbolic relations between lapse \
and shift, built from"}], "Text"],

 mathD[lentzAnsatzEq],

 Cell[TextData[{"and tied to an electromagnetic / plasma source via \
Einstein\[Dash]Maxwell\[Dash]plasma equations.  Whether the original \
construction is genuinely positive-energy is debated in the \
literature."}], "Text"],

 Cell[TextData[{"Each lives in ",
   StyleBox["WarpBubbleSim.wl", "Input"],
   " under the obvious function name (",
   StyleBox["AlcubierreMetric", "Input"], ", ",
   StyleBox["NatarioMetric", "Input"], ", ",
   StyleBox["VanDenBroeckMetric", "Input"], ", ",
   StyleBox["WhiteToroidalMetric", "Input"], ", ",
   StyleBox["BobrickMartireMetric", "Input"], ", ",
   StyleBox["LentzMetric", "Input"], "), all returning a \
4\[Times]4 symbolic ",
   mathI[ggMnLow], ".  The figure below visualises the geometric \
\"footprint\" of each \[LongDash] for Alcubierre, the analytic \[Rho]; \
for the others, an analogous shape-function-driven proxy."}], "Text"],

 inputCell["PlotMetricComparison[Range -> 2.5, Grid -> 90]"],

 imgCell[loadPng["metric_comparison.png", 760],
   "Figure 8. The six warp-drive metric families side by side.  All \
panels are slices through the y-x plane.  Top row: Alcubierre's \
energy density, Nat\[AAcute]rio's shift-derivative proxy (no \
expansion/contraction regions, just a wall), Van Den Broeck's \
pocket factor.  Bottom row: White's toroidal shape, the \
Bobrick\[Dash]Martire shell function (a positive-energy double-shell \
at R_inner = 1, R_outer = 2), and Lentz's soliton intensity profile."],

 (* ----------- Section 9: Energy conditions ---------- *)

 Cell["9. Why this is hard: energy conditions", "Section"],

 Cell[TextData[{"The classical energy conditions are coordinate-\
invariant statements about which kinds of stress-energy are \
\"reasonable\".  Four of them are standard.  The ",
   StyleBox["weak energy condition (WEC)", FontWeight -> Bold],
   " says any timelike observer measures non-negative energy density,"}],
   "Text"],

 mathD[wecEq],

 Cell[TextData[{"The ",
   StyleBox["null energy condition (NEC)", FontWeight -> Bold],
   " is the same statement specialised to null observers (limiting \
behaviour as ", mathI[RowBox[{"v", "\[Rule]", "c"}]], "):"}], "Text"],

 mathD[necEq],

 Cell[TextData[{"The ",
   StyleBox["strong energy condition (SEC)", FontWeight -> Bold],
   " controls the trace-reversed combination,"}], "Text"],

 mathD[secEq],

 Cell[TextData[{"with ", mathI[RowBox[{"T", "=",
     RowBox[{SuperscriptBox["g", RowBox[{"\[Mu]", "\[Nu]"}]], " ", TMnLow}]}]],
   " the trace of the stress-energy.  The SEC is what enforces \
gravitational attraction (the focusing theorem); cosmologists \
routinely violate it with positive cosmological constant or \
inflation.  Finally the ",
   StyleBox["dominant energy condition (DEC)", FontWeight -> Bold],
   " is the statement that energy never propagates faster than \
light: WEC plus the requirement that ",
   mathI[RowBox[{"-", SubsuperscriptBox["T", "\[Nu]", "\[Mu]"], " ",
     SuperscriptBox["u", "\[Nu]"]}]],
   " be a non-spacelike (timelike or null) 4-vector for every \
timelike ", mathI["u"], "."}], "Text"],

 Cell[TextData[{"For ordinary matter \[LongDash] dust, photons, \
fluids, electromagnetic fields \[LongDash] all four are obviously \
satisfied.  For an Alcubierre bubble at superluminal speed all \
four are ",
   StyleBox["violated", FontWeight -> Bold],
   ", which is the rigorous statement of \"requires exotic \
matter\".  The package ships sampling-based checkers for all four \
conditions; applied to a wall point of the v=c Alcubierre bubble \
(the option ",
   StyleBox["EvalAt", "Input"],
   " substitutes coordinates after symbolic differentiation so the \
random observer sampler operates on a numerical metric):"}],
      "Text"],

 inputCell[
"SeedRandom[42];
g = AlcubierreMetric[{1.0, 1.0, 8.0}, {t, x, y, z}];
CheckAllConditions[g, {t, x, y, z},
   EvalAt -> {t -> 0., x -> 1.0, y -> 0.2, z -> 0.}]
(*
  <| WEC -> {False, -0.288},
     NEC -> {False, -0.403},
     SEC -> {False, -0.735},
     DEC -> {False, -Infinity} |>
  All four classical conditions are violated.  DEC reports
  -Infinity because CheckDEC short-circuits when WEC has already
  failed (a non-spacelike energy current presupposes non-negative
  energy density).  This is what \"exotic matter\" means in cold
  equation form.
*)"],

 Cell[TextData[{"Bobrick & Martire's 2021 paper showed something \
striking about this picture: drop the requirement of superluminal \
speed and the energy conditions can be respected.  Subluminal \
positive-energy warp drives ",
   StyleBox["exist", FontSlant -> Italic],
   ".  They still cost an enormous amount of energy (orders of \
magnitude above any human energy budget), but the matter content \
is in principle sourceable by ordinary fields:"}], "Text"],

 inputCell[
"SeedRandom[42];
gBM = BobrickMartireMetric[{0.5, 1.0, 2.0, 0.1, 5.0},
                            {t, x, y, z}];   (* v=0.5c, shell *)
CheckAllConditions[gBM, {t, x, y, z},
   EvalAt -> {t -> 0., x -> 1.5, y -> 0., z -> 0.}]   (* in the shell *)
(*
  <| WEC -> {True,  +0.0032},
     NEC -> {True,  +0.00046},
     SEC -> {False, -0.00056},
     DEC -> {True,  -1.16e-5} |>
  WEC, NEC, and DEC are satisfied -- positive-energy matter could
  in principle source this geometry.  SEC is violated, but SEC is
  the condition that enforces gravitational attraction; positive
  cosmological constant and inflation violate it routinely, so an
  SEC-only violation is qualitatively different from the Alcubierre
  case where every observer-independent condition fails.
*)"],

 Cell[TextData[{"This is the clearest path forward in the \
literature: instead of trying to engineer exotic matter, find \
geometries which require none, even if the resulting drive is \
slower-than-light.  None of this rescues warp drives from being \
hypothetical objects \[LongDash] even subluminal positive-energy \
constructions demand stress-energy distributions far outside any \
matter humans know how to assemble, and the energy budgets remain \
many orders of magnitude beyond global energy production.  The \
value of writing the metrics down is not that they are \
buildable, but that they sharpen what general relativity does and \
does not forbid: causality and the field equations alone do not \
rule out faster-than-light effective travel; what rules it out, in \
practice, is the matter sector."}], "Text"],

 (* ----------- Section 10: Implementation notes ---------- *)

 Cell["10. Implementation notes", "Section"],

 Cell[TextData[{
   "\[Bullet] The Wolfram port is a side-by-side translation of a \
~5\[ThinSpace]900-line Python package.  Wolfram's symbolic ",
   StyleBox["D[\[CenterEllipsis]]", "Input"],
   " replaces the Python finite-difference Christoffel/Riemann/Ricci \
loops, which compresses the GR module from ~2\[ThinSpace]100 lines \
to ~140 and removes a step-size hyperparameter.\n",
   "\[Bullet] Each metric is exposed as a function ",
   StyleBox["<Name>Metric[params, coords]", "Input"],
   " that returns a 4\[Times]4 symbolic ", mathI[ggMnLow],
   ".  Mix and match: pass any of them through ",
   StyleBox["EinsteinTensor", "Input"], ", ",
   StyleBox["KretschmannScalar", "Input"], ", ",
   StyleBox["IntegrateGeodesic", "Input"], ", or ",
   StyleBox["CheckAllConditions", "Input"], ".\n",
   "\[Bullet] Geodesics use ",
   StyleBox["NDSolve", "Input"],
   " against a precomputed Christoffel-symbol expression.  The \
right-hand side gets compiled once and re-evaluated cheaply at every \
step \[LongDash] no per-step finite differencing.\n",
   "\[Bullet] All animations in this notebook are pure Wolfram (no \
Python).  They use the analytic Alcubierre formulas where possible, \
and a simple Euler advection of the shift field for the geodesic \
sweep.\n",
   "\[Bullet] Tests in ", StyleBox["Wolfram/Tests.wls", "Input"],
   " verify the Minkowski limit, the unit determinant of the \
Alcubierre metric, the ADM round-trip, and that flat-space \
Christoffel symbols vanish."}], "Text"],

 (* ----------- Section 11: Conclusions ---------- *)

 Cell["11. What I take away", "Section"],

 Cell[TextData[{
   "\[Bullet] ",
   StyleBox["Symbolic GR is enormously practical in Wolfram.",
            FontWeight -> Bold],
   " The whole tensor calculus chain \[LongDash] metric, \
Christoffel, Riemann, Ricci, Einstein, Weyl, Kretschmann \[LongDash] \
fits in 140 lines of Wolfram against ~1\[ThinSpace]100 lines of \
NumPy with finite differences.  The symbolic answer is exact, valid \
at every point of spacetime simultaneously, and trivially \
substituted into for numerical work.\n",
   "\[Bullet] ", StyleBox["Exotic matter is real, mathematically.",
                          FontWeight -> Bold],
   " The Alcubierre energy density is closed-form negative; the \
classical energy conditions are violated unambiguously.  Whether \
that exotic matter is physically realisable is an open question \
\[LongDash] Casimir-effect calculations and squeezed-light \
experiments do produce regions of locally negative energy density, \
but at densities and volumes ~30 orders of magnitude below what a \
useful warp drive needs.\n",
   "\[Bullet] ", StyleBox["Subluminal drives are positive-energy.",
                          FontWeight -> Bold],
   " Bobrick\[Dash]Martire's 2021 result is the most encouraging \
recent development: the ban on warp drives only applies above ",
   mathI[RowBox[{vsBoxed, "=", "c"}]],
   ".  Below it, perfectly normal matter can in principle do the \
job.  The price is enormous \[LongDash] still much more than any \
existing energy source \[LongDash] but it is finite, positive, and \
not magic.\n",
   "\[Bullet] ", StyleBox["Wolfram is a great teaching tool here.",
                          FontWeight -> Bold],
   " A full pipeline (define metric, compute Einstein tensor, plot \
energy density, integrate geodesic, animate) fits on a single \
laptop screen.  The same student who would struggle to set up \
Christoffel symbols on paper can produce Figure 1 in two minutes \
in a notebook.  This is the workflow GR education \
ought to be using by default."}], "Text"],

 (* ----------- Section 12: References ---------- *)

 Cell["12. References", "Section"],

 Cell[TextData[{
   "\[Bullet] M. Alcubierre (1994), \"The warp drive: hyper-fast \
travel within general relativity\", ",
   StyleBox["Class. Quantum Grav.", FontSlant -> Italic],
   " ", StyleBox["11", FontWeight -> Bold], ", L73.  ",
   ButtonBox["doi:10.1088/0264-9381/11/5/001",
     BaseStyle -> "Hyperlink",
     ButtonData -> {URL["https://doi.org/10.1088/0264-9381/11/5/001"],
                    None}], ".\n",
   "\[Bullet] J. Nat\[AAcute]rio (2002), \"Warp drive with zero \
expansion\", ",
   StyleBox["Class. Quantum Grav.", FontSlant -> Italic], " ",
   StyleBox["19", FontWeight -> Bold], ", 1157.  ",
   ButtonBox["arXiv:gr-qc/0110086",
     BaseStyle -> "Hyperlink",
     ButtonData -> {URL["https://arxiv.org/abs/gr-qc/0110086"], None}],
   ".\n",
   "\[Bullet] C. Van Den Broeck (1999), \"A warp drive with more \
reasonable total energy requirements\", ",
   StyleBox["Class. Quantum Grav.", FontSlant -> Italic], " ",
   StyleBox["16", FontWeight -> Bold], ", 3973.  ",
   ButtonBox["arXiv:gr-qc/9905084",
     BaseStyle -> "Hyperlink",
     ButtonData -> {URL["https://arxiv.org/abs/gr-qc/9905084"], None}],
   ".\n",
   "\[Bullet] H. White (2011), \"Warp Field Mechanics 101\", AIAA \
paper, NASA-JSC.\n",
   "\[Bullet] A. Bobrick & G. Martire (2021), \"Introducing Physical \
Warp Drives\", ",
   StyleBox["Class. Quantum Grav.", FontSlant -> Italic], " ",
   StyleBox["38", FontWeight -> Bold], ", 105009.  ",
   ButtonBox["arXiv:2102.06824",
     BaseStyle -> "Hyperlink",
     ButtonData -> {URL["https://arxiv.org/abs/2102.06824"], None}],
   ".\n",
   "\[Bullet] E. W. Lentz (2021), \"Breaking the warp barrier: \
hyper-fast solitons in Einstein\[Dash]Maxwell\[Dash]plasma theory\", ",
   StyleBox["Class. Quantum Grav.", FontSlant -> Italic], " ",
   StyleBox["38", FontWeight -> Bold], ", 075015.  ",
   ButtonBox["arXiv:2006.07125",
     BaseStyle -> "Hyperlink",
     ButtonData -> {URL["https://arxiv.org/abs/2006.07125"], None}],
   ".\n",
   "\[Bullet] L. H. Ford & T. A. Roman (1996), \"Quantum field \
theory constrains traversable wormhole geometries\", ",
   StyleBox["Phys. Rev. D", FontSlant -> Italic], " ",
   StyleBox["53", FontWeight -> Bold], ", 5496.  ",
   ButtonBox["arXiv:gr-qc/9510071",
     BaseStyle -> "Hyperlink",
     ButtonData -> {URL["https://arxiv.org/abs/gr-qc/9510071"], None}],
   ".\n",
   "\[Bullet] M. J. Pfenning & L. H. Ford (1997), \"The unphysical \
nature of warp drive\", ",
   StyleBox["Class. Quantum Grav.", FontSlant -> Italic], " ",
   StyleBox["14", FontWeight -> Bold], ", 1743.  ",
   ButtonBox["arXiv:gr-qc/9702026",
     BaseStyle -> "Hyperlink",
     ButtonData -> {URL["https://arxiv.org/abs/gr-qc/9702026"], None}],
   ".\n",
   "\[Bullet] S. Finazzi, S. Liberati & C. Barcel\[OAcute] (2009), \
\"Semiclassical instability of dynamical warp drives\", ",
   StyleBox["Phys. Rev. D", FontSlant -> Italic], " ",
   StyleBox["79", FontWeight -> Bold], ", 124017.  ",
   ButtonBox["arXiv:0904.0141",
     BaseStyle -> "Hyperlink",
     ButtonData -> {URL["https://arxiv.org/abs/0904.0141"], None}],
   ".\n",
   "\[Bullet] Source code (Wolfram + Python ports): ",
   ButtonBox["github.com/mthiel74/WarpDrives",
     BaseStyle -> "Hyperlink",
     ButtonData -> {URL["https://github.com/mthiel74/WarpDrives"], None}],
   "."}],
      "Text"]
};

(* ---------- write the notebook ----------- *)

nb = Notebook[cells,
  WindowSize -> {1100, 800},
  WindowMargins -> {{Automatic, 0}, {Automatic, 0}},
  FrontEndVersion -> "15.0 for Mac OS X ARM (64-bit)",
  StyleDefinitions -> "Default.nb",
  Saveable -> True];

outNb = FileNameJoin[{hereDir, "warpdrives.nb"}];

Print["[notebook] writing -> ", outNb];
Export[outNb, nb];
Print["[notebook] done; size = ",
      Round[FileByteCount[outNb]/1024], " KB"];
