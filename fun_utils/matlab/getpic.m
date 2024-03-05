a = test_distance(2901:2910);
x = 0:10:90;
b = cl(2901:2910);
plot(x,a,'LineWidth',2,'Color','blue');
hold on;
plot(x,b,'LineWidth',2,'Color','red');
xlabel('ms');
set(gca,'FontSize',14);
