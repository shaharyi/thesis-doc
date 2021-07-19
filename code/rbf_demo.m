x = -3:.1:3;
a = radbas(x);
a2 = radbas(x-1.5);
a3 = radbas(x+2);
a4 = a + a2*1 + a3*0.5;
set(gca,'FontSize',16);
plot(x,a,'r--',x,a2,'g--',x,a3,'k--',x,a4,'m-','LineWidth',3);
legend({'r_1','r_2','r_3','r_1+r_2+0.5*r_3'});
