# The deck and the paper share one setup, so a figure looks the same and a
# statistic is computed the same way in both.
source(file.path("..", "R", "swarm_setup.R"))

# The deck is projected, so it needs larger type than a page does.
theme_swarm <- local({
  base_theme <- theme_swarm
  function(base = 15) base_theme(base)
})
SURFACE <- "#fcfcfb"
