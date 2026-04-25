(* ::Package:: *)

(* ::Title:: *)
(*WarpBubbleSim`Geodesics`*)

(* ::Text:: *)
(*Geodesic integration in a warp-bubble spacetime.*)
(*Equations of motion:*)
(*    dx^\[Mu]/d\[Lambda] = u^\[Mu]*)
(*    du^\[Mu]/d\[Lambda] = -\[CapitalGamma]^\[Mu]_{\[Alpha]\[Beta]} u^\[Alpha] u^\[Beta]*)

BeginPackage["WarpBubbleSim`Geodesics`", {"WarpBubbleSim`Tensors`"}];

GeodesicRHS::usage =
  "GeodesicRHS[christoffelArray][state, \[Lambda]] returns {dx/d\[Lambda], du/d\[Lambda]} given the current state.";

NormalizeVelocity::usage =
  "NormalizeVelocity[u, g, \"Timelike\"|\"Null\"] rescales a 4-velocity to the proper norm.";

CreateInitialVelocity::usage =
  "CreateInitialVelocity[g0, direction, speed, \"Timelike\"|\"Null\"] builds an initial \
4-velocity at a point with local metric g0.";

IntegrateGeodesic::usage =
  "IntegrateGeodesic[metricFunc, x0, u0, {\[Lambda]min, \[Lambda]max}] integrates a geodesic \
using NDSolve.  metricFunc must accept four symbolic arguments and return a 4x4 matrix.";

IntegrateGeodesicBundle::usage =
  "IntegrateGeodesicBundle[metricFunc, x0, velocities, span] integrates a list of \
geodesics sharing the same starting point.";

Options[IntegrateGeodesic] = {Method -> "StiffnessSwitching",
                              MaxSteps -> 10^5,
                              AccuracyGoal -> 8,
                              PrecisionGoal -> 6};

Begin["`Private`"];

NormalizeVelocity[u_?VectorQ, g_?MatrixQ, "Timelike"] :=
  Module[{n = u . g . u},
    If[n >= 0, Message[NormalizeVelocity::notTimelike]; u,
      u/Sqrt[-n]]];
NormalizeVelocity[u_?VectorQ, _?MatrixQ, "Null"] := u;
NormalizeVelocity::notTimelike = "Vector not timelike.";

CreateInitialVelocity[g0_?MatrixQ, direction_?VectorQ, speed_, type_:"Timelike"] :=
  Module[{n = direction/Norm[direction], uRaw, disc, a, b, c, v},
    Switch[type,
      "Timelike",
        uRaw = Prepend[speed n, 1.];
        NormalizeVelocity[uRaw, g0, "Timelike"],
      "Null",
        a = Sum[g0[[1 + i, 1 + j]] n[[i]] n[[j]], {i, 3}, {j, 3}];
        b = 2 Sum[g0[[1, 1 + i]] n[[i]], {i, 3}];
        c = g0[[1, 1]];
        disc = b^2 - 4 a c;
        If[disc < 0, Message[CreateInitialVelocity::noNull]; Return[$Failed]];
        v = (-b + Sqrt[disc])/(2 a);
        Prepend[v n, 1.]]];
CreateInitialVelocity::noNull = "Cannot construct null vector in this direction.";

IntegrateGeodesic[metricFunc_, x0_?VectorQ, u0_?VectorQ,
                  {lamMin_?NumericQ, lamMax_?NumericQ}, opts : OptionsPattern[]] :=
  Module[{t, x, y, z, lam, coordSym, coordsF, chris, chrisN,
          T, X, Y, Z, U0, U1, U2, U3, rhs, sol, eqs, ics},
    coordSym = {t, x, y, z};
    (* Precompute symbolic Christoffel once *)
    chris = ChristoffelSymbols[metricFunc @@ coordSym, coordSym];
    (* Turn into compiled numerical function (t,x,y,z) -> 4x4x4 array *)
    chrisN = With[{expr = chris, vars = coordSym},
                  Function @@ {vars, expr}];

    (* State:  T[\[Lambda]], X[\[Lambda]], Y[\[Lambda]], Z[\[Lambda]] and their derivatives *)
    eqs = {
      T'[lam]  == U0[lam],
      X'[lam]  == U1[lam],
      Y'[lam]  == U2[lam],
      Z'[lam]  == U3[lam],
      U0'[lam] == -Sum[chrisN[T[lam], X[lam], Y[lam], Z[lam]][[1, a, b]] *
                         ({U0[lam], U1[lam], U2[lam], U3[lam]}[[a]]) *
                         ({U0[lam], U1[lam], U2[lam], U3[lam]}[[b]]),
                       {a, 4}, {b, 4}],
      U1'[lam] == -Sum[chrisN[T[lam], X[lam], Y[lam], Z[lam]][[2, a, b]] *
                         ({U0[lam], U1[lam], U2[lam], U3[lam]}[[a]]) *
                         ({U0[lam], U1[lam], U2[lam], U3[lam]}[[b]]),
                       {a, 4}, {b, 4}],
      U2'[lam] == -Sum[chrisN[T[lam], X[lam], Y[lam], Z[lam]][[3, a, b]] *
                         ({U0[lam], U1[lam], U2[lam], U3[lam]}[[a]]) *
                         ({U0[lam], U1[lam], U2[lam], U3[lam]}[[b]]),
                       {a, 4}, {b, 4}],
      U3'[lam] == -Sum[chrisN[T[lam], X[lam], Y[lam], Z[lam]][[4, a, b]] *
                         ({U0[lam], U1[lam], U2[lam], U3[lam]}[[a]]) *
                         ({U0[lam], U1[lam], U2[lam], U3[lam]}[[b]]),
                       {a, 4}, {b, 4}]};
    ics = {T[lamMin] == x0[[1]], X[lamMin] == x0[[2]],
           Y[lamMin] == x0[[3]], Z[lamMin] == x0[[4]],
           U0[lamMin] == u0[[1]], U1[lamMin] == u0[[2]],
           U2[lamMin] == u0[[3]], U3[lamMin] == u0[[4]]};
    sol = NDSolve[Join[eqs, ics],
                  {T, X, Y, Z, U0, U1, U2, U3},
                  {lam, lamMin, lamMax},
                  Sequence @@ FilterRules[{opts}, Options[NDSolve]]];
    <|"Solution" -> First[sol],
      "LambdaRange" -> {lamMin, lamMax},
      "Coords" -> Function[l, {T[l], X[l], Y[l], Z[l]} /. First[sol]],
      "Velocity" -> Function[l, {U0[l], U1[l], U2[l], U3[l]} /. First[sol]]|>];

IntegrateGeodesicBundle[metricFunc_, x0_?VectorQ, us_List, span_List, opts : OptionsPattern[]] :=
  Table[IntegrateGeodesic[metricFunc, x0, u, span, opts], {u, us}];

End[];
EndPackage[];
