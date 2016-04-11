options <- commandArgs(trailingOnly = TRUE)
name = options[1]
print(name)
trace = read.table(paste('/Users/omidi/Dropbox/PycharmProjects/CellLineageTracking/Tracking/', name, sep=''), row.names = 2)

intensity = c()
area = c()
pos = matrix(0, nr=dim(trace)[1], nc=2)
for(index in seq(1, dim(trace)[1])) {
  time = trace[index, 1]
  df = read.csv(paste('~/image_analysis/output_',time, '.csv', sep=''), row.names=1)
  intensity = c(intensity, df[rownames(trace)[index], 'intensity'])
  area = c(area, df[rownames(trace)[index], 'area'])
  pos[index, ] = c(df[rownames(trace)[index], 'x'],
                   df[rownames(trace)[index], 'y'])
}

pdf(paste(name, 'pdf', sep='.'), height = 10, width = 5)
par(mfrow=c(3,1))
plot(trace[,1], intensity, type= 'l', xlab="time")
plot(trace[,1], area, type= 'l', xlab="time")
plot(pos[,1], pos[,2], xlim=c(0,550), ylim=c(0,550), type = 'l', xlab="X", ylab = "Y")
dev.off()
