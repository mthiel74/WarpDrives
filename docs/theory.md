# Theory and Mathematical Foundations

This document describes the mathematical foundations and conventions used in WarpBubbleSim.

## Conventions

### Signature and Units
- **Metric signature**: (-,+,+,+) (mostly plus)
- **Index ordering**: (t,x,y,z) = (0,1,2,3)
- **Units**: G = c = 1 (geometric units)

### Einstein Equations
The Einstein field equations relate geometry to matter:

$$G_{\mu\nu} = 8\pi T_{\mu\nu}$$

Therefore:
$$T_{\mu\nu} = \frac{1}{8\pi} G_{\mu\nu}$$

## ADM Formalism

The 3+1 (ADM) decomposition writes the 4-metric in terms of:
- **Lapse** $\alpha$: relates proper time to coordinate time
- **Shift** $\beta^i$: relates spatial coordinates between slices
- **Spatial metric** $\gamma_{ij}$: metric on spatial hypersurfaces

The line element:
$$ds^2 = -\alpha^2 dt^2 + \gamma_{ij}(dx^i + \beta^i dt)(dx^j + \beta^j dt)$$

Expanding:
$$ds^2 = (-\alpha^2 + \beta_i\beta^i)dt^2 + 2\beta_i dx^i dt + \gamma_{ij}dx^i dx^j$$

where $\beta_i = \gamma_{ij}\beta^j$.

### 4-Metric Components
$$g_{00} = -\alpha^2 + \beta_i\beta^i$$
$$g_{0i} = \beta_i$$
$$g_{ij} = \gamma_{ij}$$

### Inverse Metric
$$g^{00} = -1/\alpha^2$$
$$g^{0i} = \beta^i/\alpha^2$$
$$g^{ij} = \gamma^{ij} - \beta^i\beta^j/\alpha^2$$

## Christoffel Symbols

The Christoffel symbols of the second kind:
$$\Gamma^\mu_{\alpha\beta} = \frac{1}{2}g^{\mu\nu}(\partial_\alpha g_{\beta\nu} + \partial_\beta g_{\alpha\nu} - \partial_\nu g_{\alpha\beta})$$

These are symmetric in the lower indices: $\Gamma^\mu_{\alpha\beta} = \Gamma^\mu_{\beta\alpha}$

## Curvature Tensors

### Riemann Tensor
$$R^\rho_{\sigma\mu\nu} = \partial_\mu\Gamma^\rho_{\nu\sigma} - \partial_\nu\Gamma^\rho_{\mu\sigma} + \Gamma^\rho_{\mu\lambda}\Gamma^\lambda_{\nu\sigma} - \Gamma^\rho_{\nu\lambda}\Gamma^\lambda_{\mu\sigma}$$

### Ricci Tensor
$$R_{\mu\nu} = R^\rho_{\mu\rho\nu}$$

### Ricci Scalar
$$R = g^{\mu\nu}R_{\mu\nu}$$

### Einstein Tensor
$$G_{\mu\nu} = R_{\mu\nu} - \frac{1}{2}g_{\mu\nu}R$$

### Weyl Tensor (4D)
$$C_{\rho\sigma\mu\nu} = R_{\rho\sigma\mu\nu} - \frac{1}{2}(g_{\rho\mu}R_{\sigma\nu} - g_{\rho\nu}R_{\sigma\mu} + g_{\sigma\nu}R_{\rho\mu} - g_{\sigma\mu}R_{\rho\nu}) + \frac{R}{6}(g_{\rho\mu}g_{\sigma\nu} - g_{\rho\nu}g_{\sigma\mu})$$

## Curvature Invariants

### Kretschmann Scalar
$$K = R_{\alpha\beta\gamma\delta}R^{\alpha\beta\gamma\delta}$$

Measures the "strength" of spacetime curvature.
- For Schwarzschild: $K = 48M^2/r^6$
- For flat spacetime: $K = 0$

### Euler Density (Gauss-Bonnet)
$$E_4 = R_{\alpha\beta\gamma\delta}R^{\alpha\beta\gamma\delta} - 4R_{\mu\nu}R^{\mu\nu} + R^2$$

## Extrinsic Curvature

The extrinsic curvature $K_{ij}$ of spatial hypersurfaces:
$$K_{ij} = \frac{1}{2\alpha}(D_i\beta_j + D_j\beta_i - \partial_t\gamma_{ij})$$

where $D_i$ is the covariant derivative with respect to $\gamma_{ij}$.

For Alcubierre-type metrics with $\alpha=1$, $\gamma_{ij}=\delta_{ij}$, and time-independent shift:
$$K_{ij} = \frac{1}{2}(\partial_i\beta_j + \partial_j\beta_i)$$

### Expansion Scalar
$$\theta = K = \gamma^{ij}K_{ij}$$

- $\theta > 0$: expansion (space stretching)
- $\theta < 0$: contraction (space compressing)

## Energy Conditions

### Weak Energy Condition (WEC)
$$T_{\mu\nu}u^\mu u^\nu \geq 0$$ for all timelike $u^\mu$

Physically: energy density is non-negative for all observers.

### Null Energy Condition (NEC)
$$T_{\mu\nu}k^\mu k^\nu \geq 0$$ for all null $k^\mu$

The weakest of the classical conditions.

### Strong Energy Condition (SEC)
$$(T_{\mu\nu} - \frac{1}{2}Tg_{\mu\nu})u^\mu u^\nu \geq 0$$ for timelike $u^\mu$

Implies gravity is attractive.

### Dominant Energy Condition (DEC)
- WEC is satisfied
- For any timelike $u^\mu$, $-T^\mu_\nu u^\nu$ is non-spacelike

Implies energy doesn't flow faster than light.

## Geodesic Equations

Geodesics satisfy:
$$\frac{d^2x^\mu}{d\lambda^2} + \Gamma^\mu_{\alpha\beta}\frac{dx^\alpha}{d\lambda}\frac{dx^\beta}{d\lambda} = 0$$

Written as first-order system:
$$\frac{dx^\mu}{d\lambda} = u^\mu$$
$$\frac{du^\mu}{d\lambda} = -\Gamma^\mu_{\alpha\beta}u^\alpha u^\beta$$

### Normalization
- Timelike: $g_{\mu\nu}u^\mu u^\nu = -1$
- Null: $g_{\mu\nu}k^\mu k^\nu = 0$

---

## Alcubierre Metric

Reference: Alcubierre, M. (1994). Class. Quantum Grav. 11, L73.

### ADM Form
$$\alpha = 1, \quad \gamma_{ij} = \delta_{ij}, \quad \beta^i = (-v_s f(r_s), 0, 0)$$

### Line Element
$$ds^2 = -dt^2 + (dx - v_s f(r_s) dt)^2 + dy^2 + dz^2$$

### Definitions
- Bubble center: $x_s(t) = x_0 + v_s t$
- Distance from center: $r_s = \sqrt{(x-x_s)^2 + y^2 + z^2}$
- Shape function: $f(r_s) \approx 1$ inside, $f(r_s) \approx 0$ outside

### Shape Functions
**Alcubierre's original (tanh)**:
$$f(r) = \frac{\tanh(\sigma(r+R)) - \tanh(\sigma(r-R))}{2\tanh(\sigma R)}$$

**Gaussian**:
$$f(r) = \exp\left(-\frac{r^2}{2R^2\sigma^2}\right)$$

**Compact polynomial (C²)**:
$$f(r) = \begin{cases}(1 - (r/R_{eff})^2)^3 & r < R_{eff}\\ 0 & r \geq R_{eff}\end{cases}$$

### Energy Density (Eulerian Observer)
$$\rho = -\frac{v_s^2}{32\pi}\left(\frac{df}{dr_s}\right)^2\frac{y^2+z^2}{r_s^2}$$

Key properties:
- Negative in the wall region (exotic matter required)
- Zero at bubble center and far away
- Scales as $v^2$

### Expansion Scalar
$$\theta = v_s\frac{x-x_s}{r_s}\frac{df}{dr_s}$$

- Expansion ($\theta > 0$) behind the bubble
- Contraction ($\theta < 0$) in front of the bubble

---

## Natário Metric

Reference: Natário, J. (2002). Class. Quantum Grav. 19, 1157.

### Expansion-Free Condition
$$\nabla \cdot \vec{\beta} = \partial_i\beta^i = 0$$

This eliminates the expansion/contraction regions.

### Construction
The shift can be written as curl of a vector potential:
$$\vec{\beta} = \nabla \times \vec{A}$$

which automatically ensures divergence-free.

---

## Van Den Broeck Metric

Reference: Van Den Broeck, C. (1999). Class. Quantum Grav. 16, 3973.

### Pocket Geometry
$$ds^2 = -dt^2 + B(r)^2[(dx - v_s f dt)^2 + dy^2 + dz^2]$$

where $B(r)$ is an expansion factor:
- $B \gg 1$ inside (large internal volume)
- $B = 1$ outside (normal external appearance)

### Energy Reduction
Energy scales with external radius $R_{ext}$, not internal volume.
Reduction factor $\sim (R_{ext}/R_{int})^2$.

---

## Bobrick & Martire Classification

Reference: Bobrick, A. & Martire, G. (2021). Class. Quantum Grav. 38, 105009.

### Key Results
1. Subluminal ($v < c$) positive-energy warp drives are mathematically possible
2. Energy conditions can be satisfied for sufficiently slow drives
3. General "warp shell" framework for classification

### Warp Shell Structure
- Interior: passenger region (approximately flat)
- Shell: where metric deviates from flat
- Exterior: asymptotically flat

---

## Lentz Soliton

Reference: Lentz, E. W. (2021). Class. Quantum Grav. 38, 075015.

### Einstein-Maxwell-Plasma Theory
Uses soliton-like profiles with EM + plasma field sources.

**[ASSUMPTION]** The exact parameterization in this implementation is an interpretation of Lentz's paper. See code documentation for details.
