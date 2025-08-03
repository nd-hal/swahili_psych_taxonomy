library(readr)
library(dplyr)
library(ggplot2)
library(stringr)

# Load the data
D <- read_csv("./Data/IntersectionalBias.csv")

# Process metrics 
z_DI <- D %>%
  mutate(
    Task = str_replace(Task, "Phys", ""), # Remove "Phys" from Task
    Task = str_replace(Task, "SubjectiveLit", "Literacy"), # Replace "SubjectiveLit" with "Literacy"
    Task = str_to_title(Task) # Ensure consistent title case
  ) %>%
  group_by(Task, Model, num_demos) %>%
  summarize(
    mean_DI = mean(Value, na.rm = TRUE),
    ci_lower = mean(Value, na.rm = TRUE) - qt(0.975, df = n() - 1) * (sd(Value, na.rm = TRUE) / sqrt(n())),
    ci_upper = mean(Value, na.rm = TRUE) + qt(0.975, df = n() - 1) * (sd(Value, na.rm = TRUE) / sqrt(n())),
    .groups = 'drop' # ungrouped
  )

# Function to create the plot for individual metrics 
plot_DI_individual <- function(data) {
  ggplot(data, aes(
    x = num_demos,
    y = mean_DI,
    color = Model,
    shape = Model,
    group = interaction(Model, Task)
  )) +
    geom_line(position = position_dodge(width = 0.8)) +
    geom_point(position = position_dodge(width = 0.8)) +
    geom_errorbar(aes(ymin = ci_lower, ymax = ci_upper), width = 0.8, position = position_dodge(width = 0.8)) +
    facet_wrap(~ Task, ncol = 4, scales = "free_y") +
    geom_hline(yintercept = 1.2, linetype = "dashed", color = "gray") +
    geom_hline(yintercept = 0.8, linetype = "dashed", color = "gray") +
    theme_bw() +
    theme(legend.position = "bottom") +
    scale_color_manual(
      name = "Model",
      values = c("mBERT" = "skyblue", "XLM-RoBERTa" = "darkred", "AfriBERTa" = "darkgreen", "SwahBERT" = "darkorange")
    ) +
    scale_shape_manual(
      name = "Model",
      values = c("mBERT" = 16, "XLM-RoBERTa" = 17, "AfriBERTa" = 18, "SwahBERT" = 15)
    ) +
    xlab("Number of Demographic Characteristics Considered") +
    ylab("Disparate Impact") 
  #ggtitle("Effect of Intersectionality on Disparate Impact Across Tasks")
}

# Generate and display the plot
p_DI_individual <- plot_DI_individual(z_DI)
print(p_DI_individual)

ggsave(
  filename = "./Results/Intersectionality_DI.png", 
  plot = p_DI_individual, # The plot object to save
  width = 12, # Width of the plot in inches
  height = 4, # Height of the plot in inches
  dpi = 1000 # Resolution 
)
