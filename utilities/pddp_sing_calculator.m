function [x,i]=pddp_sing_calculator(M,eps_value)
   
   [n,m]=size(M); 
   
   w=mean(M')';
 
 
   x_prev=ones(m,1);
   x=ones(m,1);
   e=ones(1,m); 
   i=0;
 
      
   while norm(x-x_prev)> eps_value || i==0
     
       i=i+1;
     
       x_prev=x;
     
       value=(M-w*e)'*((M-w*e)*x_prev);
        
       x = value/norm(value);
       
   end 
      
end