plot_time_interval <- function(d, t.min=1, t.max=NA) {
  if (is.na(t.max))   t.max = dim(d)[1] 
  if (t.min > t.max)   return -1
  time = d$V2
  f = smooth.spline(time, d$V19[t.min:t.max], spar = 0.3)
  plot(time, d$V19[t.min:t.max], ylim=range(d$V19))
  lines(f, lwd=3)
  f = smooth.spline(time, d$V5, spar= 0.3)
  lines(f, lwd=3, col='darkgrey')
}