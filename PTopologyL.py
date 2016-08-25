
from __future__ import print_function, division

import matplotlib as mpl
import numpy as np
import os.path
import xml.etree.ElementTree as ET

## colorpath = "/home/tim/Dropbox/PIK/Topology Paper/colors.soc"
colorFileName = "colors.soc"
colorpath = os.path.join(os.path.split(__file__)[0], colorFileName)

tree = ET.parse(colorpath)
root = tree.getroot()

allcolors = {}
for child in root.getchildren():
	allcolors[child.attrib['{http://openoffice.org/2000/drawing}name']] = child.attrib['{http://openoffice.org/2000/drawing}color']

# print("loading colors from %s"%colorpath)

topColors = [key for key in allcolors.keys() if key.startswith("TOP")]
topColors.sort()

topDict = {
		"TOP Default Flow" : "cDefault",
		"TOP management" :   "cMod",

## 		"TOP Upstream (borders)" : "
		"TOP Shelters (upstream)" : "cShelter",
		"TOP Glades (upstream)" : "cGlade",
		"TOP Lakes (upstream)" : "cLake",
		"TOP Sunny Upstream (remaining)" : "cSunnyUp",
		"TOP Dark Upstream" : "cDarkUp",

## 		"TOP Downstream (borders)"
		"TOP Backwaters (downstream)" : "cBackwaters",
		"TOP Sunny Downstream (remaining)" : "cSunnyDown",
		"TOP Dark Downstream" : "cDarkDown",

##   "TOP Eddies (border)"
		"TOP Sunny Eddies" : "cSunnyEddie",
		"TOP Dark Eddies" : "cDarkEddie",

##   "TOP Abysses (border)"
	      	"TOP Sunny Abysses" : "cSunnyAbyss",
		"TOP Dark Abysses" : "cDarkAbyss",

		"TOP Trenches" : "cTrench"}

# check if all necessary colors are defined
assert set(allcolors.keys()).issuperset(topDict.keys()), "some colors are not defined (color file is %s)"%colorpath

for colTop, colC in sorted(topDict.items()):
	# print("{:<35} as {:<13} : {:<7}".format(colTop, colC, allcolors[colTop]))
	exec("%s = '%s'"%(colC, allcolors[colTop]))

# print ()
# print ("added by Hand:")
cBound = "#444444"
cAreaBound = "#FFFFFF"
# for col in ["cBound", "cAreaBound"]:
	# print (" "*39 + "{:<13} :{:<7}".format(col, eval(col)))



## assert False


styleDefault = {"linewidth": 3,
		"color":cDefault,
		"arrowsize": 1.5}
styleMod1 = {	"linewidth": 1,
		"color":cMod,
		"linestyle": "dotted",
		"arrowsize": 1.5}
styleMod2 = {	"linewidth": 1,
		"color":cMod,
		"linestyle": "--",
		"arrowsize": 1.5}


stylePatch = {	"linewidth": 2,
		"closed": True,
		"fill": True,
		"edgecolor": cAreaBound,
                "alpha": 0.5}

stylePoint = dict(markersize=10)

# plot styling (:
mpl.rcParams["axes.labelsize"] = 24
mpl.rcParams["xtick.labelsize"] = 16
mpl.rcParams["ytick.labelsize"] = 16
mpl.rcParams["font.size"] = 30
## mpl.rcParams["font.style"] = "oblique"


