(* ::Package:: *)

(* ::Title:: *)
(*WarpBubbleSim`Metrics`*)

(* ::Text:: *)
(*Six warp-drive metric families, each returned as a 4x4 symbolic metric tensor*)
(*g\[Mu]\[Nu](t,x,y,z) with signature (-,+,+,+).*)

BeginPackage["WarpBubbleSim`Metrics`", {"WarpBubbleSim`ShapeFunctions`"}];

ADMToMetric::usage =
  "ADMToMetric[\[Alpha], \[Beta]vec, \[Gamma]mat] builds the 4-metric g_{\[Mu]\[Nu]} = \
{{-\[Alpha]^2+\[Beta]_i \[Beta]^i, \[Beta]_j}, {\[Beta]_i, \[Gamma]_{ij}}}.";

BubbleCenter::usage =
  "BubbleCenter[v0, x0, t] returns x0 + v0 t.";

RFromCenter::usage =
  "RFromCenter[{t,x,y,z}, v0, x0] = Sqrt[(x-x0-v0 t)^2 + y^2 + z^2].";

AlcubierreMetric::usage =
  "AlcubierreMetric[{v0, R, \[Sigma]}, coords] returns g_{\[Mu]\[Nu]} for the 1994 metric. \
coords may be symbolic {t,x,y,z} or numeric.";

NatarioMetric::usage =
  "NatarioMetric[{v0, R, \[Sigma]}, coords] returns g_{\[Mu]\[Nu]} for Nat\[AAcute]rio's \
expansion-free warp drive (vector-potential curl construction).";

VanDenBroeckMetric::usage =
  "VanDenBroeckMetric[{v0, Rext, Rint, Bint, \[Sigma], \[Sigma]B}, coords] returns \
g_{\[Mu]\[Nu]} for the Van Den Broeck pocket geometry.";

WhiteToroidalMetric::usage =
  "WhiteToroidalMetric[{v0, Rmajor, Rminor, \[Sigma]}, coords] returns g_{\[Mu]\[Nu]} \
for the toroidal energy distribution (heuristic).";

BobrickMartireMetric::usage =
  "BobrickMartireMetric[{v0, Rinner, Router, amp, \[Sigma]}, coords, positiveEnergy:True] \
returns g_{\[Mu]\[Nu]} for the subluminal warp-shell construction.";

LentzMetric::usage =
  "LentzMetric[{v0, R, amp, h}, coords] returns g_{\[Mu]\[Nu]} for Lentz's \
soliton drive (sech profile with hyperbolic lapse/shift).";

MetricRegistry::usage =
  "MetricRegistry[name] returns a function params |-> coords |-> g_{\[Mu]\[Nu]}. \
Known names: \"alcubierre\", \"natario\", \"vdbroek\", \"white\", \"bobrick_martire\", \"lentz\".";

Begin["`Private`"];

ADMToMetric[alpha_, beta_?VectorQ, gamma_?MatrixQ] :=
  Module[{betaLower = gamma . beta, g},
    g = ConstantArray[0, {4, 4}];
    g[[1, 1]] = -alpha^2 + betaLower . beta;
    g[[1, 2 ;; 4]] = betaLower;
    g[[2 ;; 4, 1]] = betaLower;
    g[[2 ;; 4, 2 ;; 4]] = gamma;
    g];

BubbleCenter[v0_, x0_, t_] := x0 + v0 t;

RFromCenter[{t_, x_, y_, z_}, v0_, x0_] :=
  Sqrt[(x - BubbleCenter[v0, x0, t])^2 + y^2 + z^2];

(* --------------------------------------------------------------- *)
(* 1. Alcubierre (1994)                                           *)
(* --------------------------------------------------------------- *)

AlcubierreMetric[{v0_, R_, sigma_}, {t_, x_, y_, z_},
                 OptionsPattern[{Shape -> "tanh", X0 -> 0}]] :=
  Module[{rs, f, shapeFn = ShapeFunction[OptionValue[Shape]], x0v = OptionValue[X0]},
    rs = RFromCenter[{t, x, y, z}, v0, x0v];
    f = shapeFn[rs, R, sigma];
    {{-(1 - v0^2 f^2), -v0 f, 0, 0},
     {-v0 f,            1,    0, 0},
     {0,                0,    1, 0},
     {0,                0,    0, 1}}];

(* --------------------------------------------------------------- *)
(* 2. Nat\[AAcute]rio (2002)  -- vector-potential construction              *)
(* --------------------------------------------------------------- *)

NatarioMetric[{v0_, R_, sigma_}, {t_, x_, y_, z_},
              OptionsPattern[{X0 -> 0}]] :=
  Module[{x0v = OptionValue[X0], rs, g, dg, dx, rho2, bx, by, bz, beta},
    dx = x - BubbleCenter[v0, x0v, t];
    rs = Sqrt[dx^2 + y^2 + z^2];
    g = TanhShape[rs, R, sigma];
    dg = D[TanhShape[r, R, sigma], r] /. r -> rs;
    rho2 = y^2 + z^2;
    (* \[Beta] = Curl of A with A = v0 g(r) (0, z, -y): divergence-free *)
    bx = -v0 (2 g + rho2 dg / rs);
    by =  v0 y dg dx / rs;
    bz =  v0 z dg dx / rs;
    beta = {bx, by, bz};
    ADMToMetric[1, beta, IdentityMatrix[3]]];

(* --------------------------------------------------------------- *)
(* 3. Van Den Broeck (1999)  -- pocket geometry                   *)
(* --------------------------------------------------------------- *)

VanDenBroeckMetric[{v0_, Rext_, Rint_, Bint_, sigma_, sigmaB_},
                   {t_, x_, y_, z_},
                   OptionsPattern[{X0 -> 0}]] :=
  Module[{x0v = OptionValue[X0], rs, f, B, B2, g},
    rs = RFromCenter[{t, x, y, z}, v0, x0v];
    f  = TanhShape[rs, Rext, sigma];
    B  = 1 + (Bint - 1) TanhShape[rs, Rint, sigmaB];
    B2 = B^2;
    {{-1 + B2 v0^2 f^2, -B2 v0 f, 0,  0},
     {-B2 v0 f,          B2,      0,  0},
     {0,                 0,      B2,  0},
     {0,                 0,       0, B2}}];

(* --------------------------------------------------------------- *)
(* 4. White toroidal (heuristic)                                   *)
(* --------------------------------------------------------------- *)

toroidalDistance[{t_, x_, y_, z_}, v0_, x0v_, Rmajor_, Rminor_] :=
  Module[{xs, rho, d},
    xs = BubbleCenter[v0, x0v, t];
    rho = Sqrt[y^2 + z^2];
    d = Sqrt[(rho - Rmajor)^2 + (x - xs)^2];
    d - Rminor];

WhiteToroidalMetric[{v0_, Rmajor_, Rminor_, sigma_}, {t_, x_, y_, z_},
                    OptionsPattern[{X0 -> 0}]] :=
  Module[{x0v = OptionValue[X0], d, f},
    d = toroidalDistance[{t, x, y, z}, v0, x0v, Rmajor, Rminor];
    f = (1 - Tanh[sigma d / Rminor])/2;
    {{-(1 - v0^2 f^2), -v0 f, 0, 0},
     {-v0 f,            1,    0, 0},
     {0,                0,    1, 0},
     {0,                0,    0, 1}}];

(* --------------------------------------------------------------- *)
(* 5. Bobrick & Martire (2021) -- warp shell, subluminal           *)
(* --------------------------------------------------------------- *)

BobrickMartireMetric[{v0_, Rinner_, Router_, amp_, sigma_}, {t_, x_, y_, z_},
                     OptionsPattern[{X0 -> 0, PositiveEnergy -> True}]] :=
  Module[{x0v = OptionValue[X0], pos = OptionValue[PositiveEnergy],
          rs, stepIn, stepOut, chi, f, lapse, correction, betaX, B2},
    If[v0 >= 1,
      Message[BobrickMartireMetric::superluminal, v0];
      Return[$Failed]];
    rs = RFromCenter[{t, x, y, z}, v0, x0v];
    stepIn  = (1 + Tanh[sigma (rs - Rinner)])/2;
    stepOut = (1 - Tanh[sigma (rs - Router)])/2;
    chi = stepIn stepOut;
    f = TanhShape[rs, Router, sigma];
    lapse = If[pos, 1 + amp chi v0^2 / 2, 1];
    correction = If[pos, 1 - amp chi, 1];
    betaX = -v0 f correction;
    B2 = If[pos, 1 + amp chi (1 - v0^2), 1];
    ADMToMetric[lapse, {betaX, 0, 0}, B2 IdentityMatrix[3]]];

BobrickMartireMetric::superluminal =
  "Subluminal Bobrick-Martire requires v0 < 1, got ``.";

(* --------------------------------------------------------------- *)
(* 6. Lentz (2021)  -- soliton                                     *)
(* --------------------------------------------------------------- *)

LentzMetric[{v0_, R_, amp_, hparam_}, {t_, x_, y_, z_},
            OptionsPattern[{X0 -> 0}]] :=
  Module[{x0v = OptionValue[X0], rs, phi, Nsq, lapse, betaMag, beta},
    rs = RFromCenter[{t, x, y, z}, v0, x0v];
    phi = Sech[rs/R];
    Nsq = 1 + amp v0^2 phi^2 Cosh[hparam];
    lapse = Sqrt[Max[Nsq, 0.01]];
    betaMag = v0 phi Tanh[hparam] (1 + amp phi);
    beta = {-betaMag, 0, 0};
    ADMToMetric[lapse, beta, IdentityMatrix[3]]];

(* --------------------------------------------------------------- *)
(* Registry                                                       *)
(* --------------------------------------------------------------- *)

MetricRegistry["alcubierre"]     = AlcubierreMetric;
MetricRegistry["natario"]        = NatarioMetric;
MetricRegistry["vdbroek"]        = VanDenBroeckMetric;
MetricRegistry["white"]          = WhiteToroidalMetric;
MetricRegistry["bobrick_martire"]= BobrickMartireMetric;
MetricRegistry["lentz"]          = LentzMetric;
MetricRegistry[_] := (Message[MetricRegistry::unknown]; $Failed);
MetricRegistry::unknown =
  "Unknown metric. Known names: alcubierre, natario, vdbroek, white, bobrick_martire, lentz.";

End[];
EndPackage[];
