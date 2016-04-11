-- Creates a large set of pentadecathlon + glider collisions.
-- Based on pd_glider.py from PLife (http://plife.sourceforge.net/).

local g = gollylib()
local gp = require "gpackage"
local gpo = require "gpackage.objects"

g.setrule("B3/S23")

local function collision(i, j)
    return gpo.pentadecathlon + gpo.glider[i + 11].t(-8 + j, -10, gp.flip)
end

local all = gp.pattern()
for i = -7, 7 do
    for j = -9, 9 do
        all = all + collision(i, j).t(100 * i, 100 * j)
	end
end
all.display("pd-glider")