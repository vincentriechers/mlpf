! 1) Settings used in the main program.
Random:setSeed = on
Main:timesAllowErrors = 5

! 2) Output settings.
Init:showChangedSettings = on
Init:showChangedParticleData = off
Next:numberCount = 10000
Next:numberShowInfo = 1
Next:numberShowProcess = 1
Next:numberShowEvent = 0

! 3) Beam parameters.
Beams:idA = 11
Beams:idB = -11
Beams:allowMomentumSpread  = off

! Vertex smearing :
Beams:allowVertexSpread = on
Beams:sigmaVertexX = 5.96e-3
Beams:sigmaVertexY = 23.8E-6
Beams:sigmaVertexZ = 0.397
Beams:sigmaTime = 10.89    !  36.3 ps

! 4) Hard process : e+e- -> gamma gamma at Ecm=91 GeV
Beams:eCM = 91.188

PromptPhoton:ffbar2gammagamma = on   ! e+e- -> gamma gamma

! Theta_min = 150 mrad  ->  pT = (Ecm/2)*sin(theta) = 45.594*sin(0.150)
PhaseSpace:pTHatMin = 6.81

PartonLevel:ISR = on
PartonLevel:FSR = on