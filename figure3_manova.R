mice_data <- read.csv('/home/guido/Data/Behavior/learned_mice_data.csv')
res.man <- manova(cbind(perf_easy, n_trials, threshold, bias, reaction_time) ~ lab, data = mice_data)
summary.aov(res.man)
