# trace generated using paraview version 5.13.3
#import paraview
#paraview.compatibility.major = 5
#paraview.compatibility.minor = 13

#### import the simple module from the paraview
from paraview.simple import *
#### disable automatic camera reset on 'Show'
paraview.simple._DisableFirstRenderCameraReset()

# create a new 'PVD Reader'
grid_datapvd = PVDReader(registrationName='grid_data.pvd', FileName='/scratch/users/kofib/DSMC_FEM_VALIDATION_P1/BEAM_LOCKIN/PHASE_0/EXAMPLE/grid_data.pvd')

# get animation scene
animationScene1 = GetAnimationScene()

# update animation scene based on data timesteps
animationScene1.UpdateAnimationUsingDataTimeSteps()

# get active view
renderView1 = GetActiveViewOrCreate('RenderView')

# show data in view
grid_datapvdDisplay = Show(grid_datapvd, renderView1, 'UnstructuredGridRepresentation')

# trace defaults for the display properties.
grid_datapvdDisplay.Representation = 'Surface'

# reset view to fit data
renderView1.ResetCamera(False, 0.9)

#changing interaction mode based on data extents
renderView1.CameraPosition = [2.0, 0.0, 20.1]
renderView1.CameraFocalPoint = [2.0, 0.0, 0.0]

# get the material library
materialLibrary1 = GetMaterialLibrary()

# update the view to ensure updated data information
renderView1.Update()

# Properties modified on renderView1
renderView1.UseColorPaletteForBackground = 0

# create a new 'Cell Data to Point Data'
cellDatatoPointData1 = CellDatatoPointData(registrationName='CellDatatoPointData1', Input=grid_datapvd)

# show data in view
cellDatatoPointData1Display = Show(cellDatatoPointData1, renderView1, 'UnstructuredGridRepresentation')

# trace defaults for the display properties.
cellDatatoPointData1Display.Representation = 'Surface'

# hide data in view
Hide(grid_datapvd, renderView1)

# update the view to ensure updated data information
renderView1.Update()

# reset view to fit data bounds
renderView1.ResetCamera(-1.0, 5.0, -2.0, 2.0, 0.0, 0.0, True, 0.9)

# create a new 'Annotate Time Filter'
annotateTimeFilter1 = AnnotateTimeFilter(registrationName='AnnotateTimeFilter1', Input=cellDatatoPointData1)

# Properties modified on annotateTimeFilter1
annotateTimeFilter1.Format = 'Time: {time:f} ms'
annotateTimeFilter1.Scale = 0.001

# show data in view
annotateTimeFilter1Display = Show(annotateTimeFilter1, renderView1, 'TextSourceRepresentation')

# update the view to ensure updated data information
renderView1.Update()

# Properties modified on annotateTimeFilter1Display
annotateTimeFilter1Display.WindowLocation = 'Upper Center'

# Properties modified on annotateTimeFilter1Display
annotateTimeFilter1Display.FontFamily = 'Times'

# Properties modified on annotateTimeFilter1Display
annotateTimeFilter1Display.Color = [0.0, 0.0, 0.0]

# set active source
SetActiveSource(cellDatatoPointData1)

# create a new 'Calculator'
calculator1 = Calculator(registrationName='Calculator1', Input=cellDatatoPointData1)

# Properties modified on calculator1
calculator1.ResultArrayName = 'Velocity'
calculator1.Function = '"f_OUTPUT[3]"*iHat+"f_OUTPUT[4]"*jHat+0*kHat'

# show data in view
calculator1Display = Show(calculator1, renderView1, 'UnstructuredGridRepresentation')

# trace defaults for the display properties.
calculator1Display.Representation = 'Surface'

# hide data in view
Hide(cellDatatoPointData1, renderView1)

# update the view to ensure updated data information
renderView1.Update()

# set scalar coloring
ColorBy(calculator1Display, ('POINTS', 'Velocity', 'Magnitude'))

# rescale color and/or opacity maps used to include current data range
calculator1Display.RescaleTransferFunctionToDataRange(True, False)

# show color bar/color legend
calculator1Display.SetScalarBarVisibility(renderView1, True)

# get color transfer function/color map for 'Velocity'
velocityLUT = GetColorTransferFunction('Velocity')

# get opacity transfer function/opacity map for 'Velocity'
velocityPWF = GetOpacityTransferFunction('Velocity')

# get 2D transfer function for 'Velocity'
velocityTF2D = GetTransferFunction2D('Velocity')

# Properties modified on velocityLUT
velocityLUT.ShowDataHistogram = 0

# Properties modified on velocityLUT
velocityLUT.ShowDataHistogram = 1

# Properties modified on velocityLUT
velocityLUT.AutomaticDataHistogramComputation = 0

# Properties modified on velocityLUT
velocityLUT.AutomaticDataHistogramComputation = 1

# Properties modified on animationScene1
animationScene1.AnimationTime = 400000.0

# get the time-keeper
timeKeeper1 = GetTimeKeeper()

# get color legend/bar for velocityLUT in view renderView1
velocityLUTColorBar = GetScalarBar(velocityLUT, renderView1)

# Properties modified on velocityLUT
velocityLUT.DataHistogramNumberOfBins = 200

# Rescale transfer function
velocityLUT.RescaleTransferFunction(0.0, 325.0)

# Rescale transfer function
velocityPWF.RescaleTransferFunction(0.0, 325.0)

# Rescale 2D transfer function
velocityTF2D.RescaleTransferFunction(0.0, 325.0, 0.0, 1.0)

# set active source
SetActiveSource(annotateTimeFilter1)

# toggle interactive widget visibility (only when running from the GUI)
ShowInteractiveWidgets(proxy=annotateTimeFilter1Display)

# toggle interactive widget visibility (only when running from the GUI)
HideInteractiveWidgets(proxy=annotateTimeFilter1Display)

# Properties modified on annotateTimeFilter1Display
annotateTimeFilter1Display.FontSize = 4

# Properties modified on annotateTimeFilter1Display
annotateTimeFilter1Display.FontSize = 40

# get layout
layout1 = GetLayout()

# layout/tab size in pixels
layout1.SetSize(1712, 1242)

# current camera placement for renderView1
renderView1.InteractionMode = '2D'
renderView1.CameraPosition = [2.0, 0.0, 13.930780379920632]
renderView1.CameraFocalPoint = [2.0, 0.0, 0.0]
renderView1.CameraParallelScale = 2.4219532516259275

# save animation
SaveAnimation(filename='/Users/kofiblake/Documents/TESTANIMATION.avi', viewOrLayout=renderView1, location=16, ImageResolution=[1712, 1240],
    FrameRate=24,
    FrameWindow=[0, 20])

#================================================================
# addendum: following script captures some of the application
# state to faithfully reproduce the visualization during playback
#================================================================

#--------------------------------
# saving layout sizes for layouts

# layout/tab size in pixels
layout1.SetSize(1712, 1242)

#-----------------------------------
# saving camera placements for views

# current camera placement for renderView1
renderView1.InteractionMode = '2D'
renderView1.CameraPosition = [2.0, 0.0, 13.930780379920632]
renderView1.CameraFocalPoint = [2.0, 0.0, 0.0]
renderView1.CameraParallelScale = 2.4219532516259275


##--------------------------------------------
## You may need to add some code at the end of this python script depending on your usage, eg:
#
## Render all views to see them appears
# RenderAllViews()
#
## Interact with the view, usefull when running from pvpython
# Interact()
#
## Save a screenshot of the active view
# SaveScreenshot("path/to/screenshot.png")
#
## Save a screenshot of a layout (multiple splitted view)
# SaveScreenshot("path/to/screenshot.png", GetLayout())
#
## Save all "Extractors" from the pipeline browser
# SaveExtracts()
#
## Save a animation of the current active view
# SaveAnimation()
#
## Please refer to the documentation of paraview.simple
## https://www.paraview.org/paraview-docs/latest/python/paraview.simple.html
##--------------------------------------------