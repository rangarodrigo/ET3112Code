x <- c(1,2,3)
print(x)

ls()

x = rnorm(100)
y = rnorm(100)

plot(x,y)

pdf("Fig.pdf")
plot(x,y, col="green")
dev.off()

M = matrix(data= c(1:10), nrow=5, ncol=2)
print(M)


