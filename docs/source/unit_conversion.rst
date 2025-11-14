===============
Unit Conversion
===============

.. _ALU_conversion_label:

In order to simplify usage, reconstructions are done using arbitrary length units (ALU).
In this system, 1 ALU can correspond to any convenient measure of distance chosen by the user.

**Default Parameter Example:**
By default, 1 ALU = spacing between detector channels.
So if we reconstruct CT data from a scanner with detector channel spacing of 0.25 mm, then ``delta_channel=1.0`` and ``1 ALU = 0.25 mm``.
And the reconstruction produces 3D JAX array ``recon`` in units of :math:`\mbox{ALU}^{-1}`.

However, this can be converted to conventional units of :math:`\mbox{mm}^{-1}` with the following scaling:

.. math::

    \mbox{(recon in units of mm$^{-1}$)} = \frac{ \mbox{recon in units of ALU$^{-1}$} }{ 0.25 \mbox{mm} / \mbox{ALU}} = \mbox{4.0*recon}


**NSI Reconstruction Example:** The NSI preprocessing functions in :ref:`PreprocessDocs` set all the parameters to the
units used by NSI which are in mm.
In this case, ``1 ALU = 1 mm`` and the reconstructions will have units of :math:`\mbox{mm}^{-1}`.

In order to convert to units of :math:`\mbox{cm}^{-1}`, we use the following scaling:

.. math::

    \mbox{(recon in units of cm$^{-1}$)} = \frac{ \mbox{recon in units of ALU$^{-1}$} }{ \mbox{1 cm} / \mbox{10 mm}} = \mbox{10 *recon}



**Emission CT Example:** Once again, we assume that the channel spacing in the detector is 0.25 mm,
and we again assume the default reconstruction parameters of ``delta_channel=1.0``.
So we have that ``1 ALU = 5 mm``.

Using this convention, the 3D array, ``recon``, will be in units of photons/AU.
However, the image can be converted to units of photons/mm using the following equation:

.. math::

    \mbox{(recon in units of photons/mm)} = \frac{ \mbox{recon in units of photons/ALU} }{ 0.25 \mbox{mm} / \mbox{ALU}} = 4.0* \mbox{recon}
