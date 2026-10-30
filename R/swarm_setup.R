# Shared R setup for every rendered document: palette, theme, data loading.
#
# Sourced by both presentation/ and paper/, so a figure looks the same and a
# statistic is computed the same way wherever it appears. The repository root
# is found by walking up for the datasets/ directory, so this works from any
# subdirectory without hardcoding "..".

suppressPackageStartupMessages({
  library(readr); library(dplyr); library(tidyr); library(ggplot2)
  library(scales)
})

find_repo_root <- function(start = getwd()) {
  dir <- normalizePath(start, winslash = "/", mustWork = FALSE)
  for (i in 1:8) {
    if (dir.exists(file.path(dir, "datasets", "summary"))) return(dir)
    parent <- dirname(dir)
    if (identical(parent, dir)) break
    dir <- parent
  }
  stop("could not locate the repository root (no datasets/summary above ",
       start, ")")
}

ROOT <- find_repo_root()
DATA <- file.path(ROOT, "datasets")

# Categorical palette, validated for colour-vision-deficiency separation and
# normal-vision separation against a light surface. Assigned in fixed order,
# never cycled. The first three clear the all-pairs floors, so any chart whose
# marks are compared pairwise (scatter, bubble) uses only those.
PAL <- c(blue = "#2a78d6", orange = "#eb6834", aqua = "#1baf7a",
         yellow = "#eda100", magenta = "#e87ba4", green = "#008300")
INK <- "#0b0b0b"; MUTED <- "#52514e"; GRID <- "#dedddA"; SURFACE <- "#ffffff"

theme_swarm <- function(base = 12) {
  theme_minimal(base_size = base, base_family = "sans") +
    theme(
      plot.background  = element_rect(fill = SURFACE, colour = NA),
      panel.background = element_rect(fill = SURFACE, colour = NA),
      panel.grid.major = element_line(colour = GRID, linewidth = 0.35),
      panel.grid.minor = element_blank(),
      axis.text   = element_text(colour = MUTED, size = base - 2),
      axis.title  = element_text(colour = MUTED, size = base - 1),
      plot.title  = element_text(colour = INK, size = base, face = "bold",
                                 hjust = 0),
      plot.subtitle = element_text(colour = MUTED, size = base - 1, hjust = 0),
      plot.caption  = element_text(colour = MUTED, size = base - 3, hjust = 0),
      legend.position = "top",
      legend.title = element_text(colour = MUTED, size = base - 2),
      legend.text  = element_text(colour = MUTED, size = base - 2),
      legend.key.height = unit(9, "pt")
    )
}

# --- data -----------------------------------------------------------------
runs <- function(key) {
  read_csv(file.path(DATA, "summary", paste0(key, "_runs.csv")),
           show_col_types = FALSE)
}
analysis <- function(name) {
  read_csv(file.path(DATA, "analysis", name), show_col_types = FALSE)
}
detail <- function(name) {
  read_csv(file.path(DATA, "detail", name), show_col_types = FALSE)
}

exp1  <- runs("exp1")  |> arrange(concurrency_G)
exp2  <- runs("exp2")
exp4  <- runs("exp4")
exp1s <- runs("exp1s") |> arrange(concurrency_G)
exp7  <- analysis("exp7_mode_check.csv")
recov <- detail("exp4_recovery.csv")
cmp1  <- analysis("engine_comparison_exp1.csv")
cmp2  <- analysis("engine_comparison_exp2.csv")
beff  <- jsonlite::fromJSON(file.path(DATA, "analysis",
                                      "exp7_effective_bandwidth.json"))

# Least-squares slope of measured p50 against per-hop delay, per depth. The
# delay sweep's headline: this should equal D.
exp2_slopes <- exp2 |>
  group_by(depth_D) |>
  summarise(
    slope = coef(lm(meas_t_token_s_p50 * 1000 ~ sweep_delay_ms))[2],
    r2    = summary(lm(meas_t_token_s_p50 * 1000 ~ sweep_delay_ms))$r.squared,
    .groups = "drop") |>
  mutate(predicted = depth_D, err_pct = 100 * (slope - predicted) / predicted)

# Ratios used in the text, computed once so prose and tables cannot disagree.
ratio_ramp   <- exp1$meas_throughput_tok_s / exp1$pred_x_of_g_tok_s
ratio_closed <- exp1$meas_throughput_tok_s / exp1$pred_x_closed_loop_tok_s
ratio_ramp_scale <- exp1s$meas_throughput_tok_s / exp1s$pred_x_of_g_tok_s

pct  <- function(x, d = 1) sprintf(paste0("%.", d, "f%%"), 100 * x)
num  <- function(x, d = 2) formatC(x, format = "f", digits = d, big.mark = ",")
