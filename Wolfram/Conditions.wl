(* ::Package:: *)

(* ::Title:: *)
(*WarpBubbleSim`Conditions`*)

(* ::Text:: *)
(*Classical energy-condition checks, done by sampling a collection of timelike*)
(*and null observers.  Returns {satisfied, minValue}.*)

BeginPackage["WarpBubbleSim`Conditions`", {"WarpBubbleSim`Energy`"}];

TimelikeVelocities::usage =
  "TimelikeVelocities[n, g] returns a list of n normalized timelike 4-velocities at a point with metric g.";

NullVectors::usage =
  "NullVectors[n, g] returns n null 4-vectors of the metric g, sampled \
uniformly on S^2 in spatial direction.  Each k satisfies g_{\[Mu]\[Nu]} k^\[Mu] k^\[Nu] = 0.";

CheckWEC::usage =
  "CheckWEC[g, coords, opts] returns {satisfied, min T_{\[Mu]\[Nu]} u^{\[Mu]} u^{\[Nu]}} over sampled timelike observers.";

CheckNEC::usage =
  "CheckNEC[g, coords, opts] returns {satisfied, min T_{\[Mu]\[Nu]} k^{\[Mu]} k^{\[Nu]}}.";

CheckSEC::usage =
  "CheckSEC[g, coords, opts] returns {satisfied, min (T_{\[Mu]\[Nu]} - (1/2) T g_{\[Mu]\[Nu]}) u^{\[Mu]} u^{\[Nu]}}.";

CheckDEC::usage =
  "CheckDEC[g, coords, opts] returns {satisfied, max g_{\[Mu]\[Nu]} J^{\[Mu]} J^{\[Nu]} with J = -T u}.";

CheckAllConditions::usage =
  "CheckAllConditions[g, coords, opts] returns an Association of {WEC, NEC, SEC, DEC}.";

EvalAt::usage =
  "EvalAt is an option for CheckWEC/NEC/SEC/DEC/CheckAllConditions.  Pass a \
list of coordinate substitution rules (e.g. EvalAt -> {t -> 0., x -> 1.}) to \
evaluate the metric and stress-energy tensor at a numerical point before \
random observer sampling.  Default None leaves both symbolic, in which case \
sampling degenerates to the static observer at coordinate vector (1,0,0,0).";

Options[CheckWEC]           = {NSamples -> 10, Tolerance -> 10^-10, EvalAt -> None};
Options[CheckNEC]           = {NSamples -> 20, Tolerance -> 10^-10, EvalAt -> None};
Options[CheckSEC]           = {NSamples -> 10, Tolerance -> 10^-10, EvalAt -> None};
Options[CheckDEC]           = {NSamples -> 10, Tolerance -> 10^-10, EvalAt -> None};
Options[CheckAllConditions] = {NSamples -> 10, Tolerance -> 10^-10, EvalAt -> None};

Begin["`Private`"];

randomUnit3[] :=
  Module[{v = RandomVariate[NormalDistribution[], 3]}, v/Norm[v]];

TimelikeVelocities[n_Integer, g_?MatrixQ] :=
  Module[{out = {}, attempts = 0, vm, dir, u, nrm},
    While[Length[out] < n && attempts < 50 n,
      attempts++;
      vm = 0.9 RandomReal[];
      dir = randomUnit3[];
      u = Prepend[vm dir, 1];
      nrm = u . g . u;
      If[NumericQ[nrm] && nrm < 0,
        AppendTo[out, Re[u/Sqrt[-nrm]]]]];
    If[out === {}, {{1., 0., 0., 0.}}, out]];

NullVectors[n_Integer, g_?MatrixQ] :=
  Module[{out = {}, attempts = 0, dir, a, b, c, disc, v, i, j},
    While[Length[out] < n && attempts < 50 n,
      attempts++;
      dir = randomUnit3[];
      a = Sum[g[[1 + i, 1 + j]] dir[[i]] dir[[j]], {i, 3}, {j, 3}];
      b = 2 Sum[g[[1, 1 + i]] dir[[i]], {i, 3}];
      c = g[[1, 1]];
      disc = b^2 - 4 a c;
      If[NumericQ[disc] && disc >= 0 && a != 0,
        v = (-b + Sqrt[disc])/(2 a);
        AppendTo[out, Re @ Prepend[v dir, 1.]]]];
    If[out === {}, {{1., 0., 0., 1.}}, out]];

(* Substitute EvalAt rule (or list of rules) into a symbolic expression. *)
applyEvalAt[expr_, None] := expr;
applyEvalAt[expr_, rule_Rule] := expr /. rule;
applyEvalAt[expr_, rules_List] := expr /. rules;

CheckWEC[g_, coords_List, OptionsPattern[]] :=
  Module[{T, vals, gEval, n = OptionValue[NSamples], tol = OptionValue[Tolerance],
          ev = OptionValue[EvalAt]},
    T = N @ applyEvalAt[StressEnergyTensor[g, coords], ev];
    gEval = N @ applyEvalAt[g, ev];
    vals = Table[u . T . u,
                 {u, Append[TimelikeVelocities[n, gEval], {1., 0, 0, 0}]}];
    {Min[vals] >= -tol, Min[vals]}];

CheckNEC[g_, coords_List, OptionsPattern[]] :=
  Module[{T, gEval, vals, n = OptionValue[NSamples], tol = OptionValue[Tolerance],
          ev = OptionValue[EvalAt]},
    T = N @ applyEvalAt[StressEnergyTensor[g, coords], ev];
    gEval = N @ applyEvalAt[g, ev];
    vals = Table[k . T . k, {k, NullVectors[n, gEval]}];
    {Min[vals] >= -tol, Min[vals]}];

CheckSEC[g_, coords_List, OptionsPattern[]] :=
  Module[{gEval, T, ginv, tr, Tsec, vals, n = OptionValue[NSamples],
          tol = OptionValue[Tolerance], ev = OptionValue[EvalAt]},
    gEval = N @ applyEvalAt[g, ev];
    T    = N @ applyEvalAt[StressEnergyTensor[g, coords], ev];
    ginv = Inverse[gEval];
    tr   = Sum[ginv[[m, nn]] T[[m, nn]], {m, 4}, {nn, 4}];
    Tsec = T - (1/2) tr gEval;
    vals = Table[u . Tsec . u,
                 {u, Append[TimelikeVelocities[n, gEval], {1., 0, 0, 0}]}];
    {Min[vals] >= -tol, Min[vals]}];

CheckDEC[g_, coords_List, OptionsPattern[]] :=
  Module[{gEval, T, ginv, Tmixed, vals, wec, n = OptionValue[NSamples],
          tol = OptionValue[Tolerance], ev = OptionValue[EvalAt]},
    wec = First @ CheckWEC[g, coords, NSamples -> n, Tolerance -> tol, EvalAt -> ev];
    If[!wec, Return[{False, -Infinity}]];
    gEval = N @ applyEvalAt[g, ev];
    T      = N @ applyEvalAt[StressEnergyTensor[g, coords], ev];
    ginv   = Inverse[gEval];
    Tmixed = ginv . T;
    vals   = Table[Module[{J = -Tmixed . u}, J . gEval . J],
                   {u, Append[TimelikeVelocities[n, gEval], {1., 0, 0, 0}]}];
    {Max[vals] <= tol, Max[vals]}];

CheckAllConditions[g_, coords_List, opts : OptionsPattern[]] :=
  <|"WEC" -> CheckWEC[g, coords, opts],
    "NEC" -> CheckNEC[g, coords, opts],
    "SEC" -> CheckSEC[g, coords, opts],
    "DEC" -> CheckDEC[g, coords, opts]|>;

End[];
EndPackage[];
