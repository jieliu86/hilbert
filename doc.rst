.. comment /*
.. comment  * @BEGIN LICENSE
.. comment  *
.. comment  * hilbert by Psi4 Developer, a plugin to:
.. comment  *
.. comment  * Psi4: an open-source quantum chemistry software package
.. comment  *
.. comment  * Copyright (c) 2007-2019 The Psi4 Developers.
.. comment  *
.. comment  * The copyrights for code used from other parties are included in
.. comment  * the corresponding files.
.. comment  *
.. comment  * This file is part of Psi4.
.. comment  *
.. comment  * Psi4 is free software; you can redistribute it and/or modify
.. comment  * it under the terms of the GNU Lesser General Public License as published by
.. comment  * the Free Software Foundation, version 3.
.. comment  *
.. comment  * Psi4 is distributed in the hope that it will be useful,
.. comment  * but WITHOUT ANY WARRANTY; without even the implied warranty of
.. comment  * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
.. comment  * GNU Lesser General Public License for more details.
.. comment  *
.. comment  * You should have received a copy of the GNU Lesser General Public License along
.. comment  * with Psi4; if not, write to the Free Software Foundation, Inc.,
.. comment  * 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
.. comment  *
.. comment  * @END LICENSE
.. comment  */

.. include:: /autodoc_abbr_options_c.rst
.. include:: /autodoc_abbr_options_plugins.rst

.. _`sec:modulename`:

Theory, Usage, and Notes
------------------------

.. codeauthor:: Psi4 Developer
.. sectionauthor:: Psi4 Developer

Casual documentation for this plugin goes here.
Uncomment the ``.. comment`` for some examples.

.. comment * this plugin solves :math:`H=F_A+W_A+F_B+W_B+V`
.. comment * reference to keyword |globals__docc| and |hilbert__print|
.. comment * returns :psivar:`CURRENT ENERGY <CURRENTENERGY>` in |kcalpermol|
.. comment * operation depends on :envvar:`OMP_NUM_THREADS`.
.. comment * operation requires external software described in :ref:`sec:interfacing`
.. comment 
.. comment .. caution:: Some features are not yet implemented.
.. comment 
.. comment    - Do not run plugin more than ten feet away from developer.
.. comment 
.. comment    - Do not run plugin on benzene dimer.


HF/DQG lambda interpolation helper
==================================

The :mod:`lambda_dqgt` helper provides a lightweight one-dimensional
post-processing workflow for HF/DQG interpolation studies.  Given T1/T2 block
matrices constructed from a DQG 2-RDM and from a Hartree-Fock 2-RDM in the same
orbital basis, it forms each interpolated block as
``(1 - lambda) * block_dqg + lambda * block_hf`` and finds the minimum-energy
feasible ``lambda`` by checking the eigenvalues of the interpolated matrices.
The helper intentionally diagonalizes the interpolated block itself rather than
interpolating endpoint eigenvalues, which is only valid when endpoint blocks are
simultaneously diagonalizable.

The helper also includes dense spin-orbital constructors for endpoint T1/T2
matrices following the same antisymmetrized formulas as the v2RDM-CASSCF
``t1.cc``/``t2.cc`` constraint maps.  Supply the v2RDM 2-RDM and either its
1-RDM or an electron count for 2-RDM contraction; the Hartree-Fock endpoint is
constructed from an idempotent Hartree-Fock 1-RDM.  The generated endpoint
blocks must use the same orbital basis before they are passed into the
interpolation optimizer.

A runnable H2/STO-3G Psi4/Hilbert example is provided in
``examples/dqgt1t2_h2_example.py``.  It follows the same pattern as the
``tests/v2rdm*`` inputs: run SCF, run :class:`hilbert.v2RDMHelper`, collect the
active-space 1-RDM/2-RDM through the Python API, build T1/T2 endpoint matrices,
and pass them to the lambda optimizer.
