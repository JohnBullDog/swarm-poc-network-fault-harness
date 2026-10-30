# Shared setup for the presentation: palette, theme, and data loading.
#
# Kept out of the .qmd so the deck reads as narrative rather than plumbing,
# and so the same styling can be reused for any other figure.

suppressPackageStartupMessages({
  library(readr); library(dplyr); library(tidyr); library(ggplot2)
  library(scales)
})

DATA <- file.path("..", "datasets")

# Categorical palette, validated for colour-vision deficiency separation and
# normal-vision separation on a light surface (OKLab dE >= 8 adjacent CVD,
# >= 15 normal). Assigned in fixed order, never cycled.
PAL <- c(blue = "#2a78d6", orange = "#eb6834", aqua = "#1baf7a",
         yellow = "#eda100", magenta = "#e87ba4", green = "#008300")
INK <- "#0b0b0b"; MUTED <- "#52514e"; GRID <- "#dedddA"; SURFACE <- "#fcfcfb"

theme_swarm <- function(base = 15) {
  theme_minimal(base_size = base, base_family = "sans") +
    theme(
      plot.background  = element_rect(fill = SURFACE, colour = NA),
      panel.background = element_rect(fill = SURFACE, colour = NA),
      panel.grid.major = element_line(colour = GRID, linewidth = 0.4),
      panel.grid.minor = element_blank(),
      axis.text   = element_text(colour = MUTED, size = base - 3),
      axis.title  = element_text(colour = MUTED, size = base - 2),
      plot.title  = element_text(colour = INK, size = base + 1, face = "bold",
                                 hjust = 0),
      plot.subtitle = element_text(colour = MUTED, size = base - 2, hjust = 0),
      plot.caption  = element_text(colour = MUTED, size = base - 4, hjust = 0),
      legend.position = "top",
      legend.title = element_blank(),
      legend.text  = element_text(colour = MUTED, size = base - 3),
      legend.key.height = unit(10, "pt")
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

# Slope of measured p50 against per-hop delay, per depth. The headline of the
# delay sweep: it should equal D.
exp2_slopes <- exp2 |>
  group_by(depth_D) |>
  summarise(
    slope = coef(lm(meas_t_token_s_p50 * 1000 ~ sweep_delay_ms))[2],
    r2    = summary(lm(meas_t_token_s_p50 * 1000 ~ sweep_delay_ms))$r.squared,
    .groups = "drop"
  ) |>
  mutate(predicted = depth_D, err_pct = 100 * (slope - predicted) / predicted)

fmt_pct <- function(x, digits = 1) sprintf(paste0("%.", digits, "f%%"), 100 * x)
