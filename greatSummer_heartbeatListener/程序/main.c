#include <reg52.h>
#define uchar unsigned char  //8λ�궨��
#define uint  unsigned int	 //16λ�궨��
#include <intrins.h>

sbit dq = P1^6;	    //����18b20 IO��
sbit beep = P3^7;   //��������� IO��

bit flag_300ms;

uchar code table_num[]="0123456789abcdefg";

sbit rs=P1^0;	 //�Ĵ���ѡ���ź� H:���ݼĴ��� L:ָ��Ĵ���
sbit rw=P1^1;	 //�Ĵ���ѡ���ź� H:���ݼĴ��� L:ָ��Ĵ���
sbit e =P1^2;	 //Ƭѡ�ź� �½��ش���

unsigned char flag_i=0,timecount=0,displayOK=0,rate=0,aa=0;

uchar rate_l,rate_h;   //������������

float time[6]={0};

uchar menu_1,menu_2;   //���ò���ʹ��
uint temperature ;

sbit clk = P1^3;	  //ds1302ʱ���߶���
sbit io =  P1^4;	  //������
sbit rst = P1^5;	  //��λ��
						//��  ��  ʱ  ��  ��  ��  ����
uchar code write_add[]={0x8a,0x8c,0x88,0x86,0x84,0x82,0x80};   //д��ַ
uchar code read_add[] ={0x81,0x83,0x85,0x87,0x89,0x8d,0x8b};   //����ַ
uchar miao,fen,shi,ri,yue,week,nian;
uchar i;

uchar flag_100ms ;

//��ʱ���� 1ms
void delay_1ms(uint q)
{
	uint i,j;
	for(i=0;i<q;i++)
		for(j=0;j<110;j++);
}

//С��ʱ
void delay_uint(uint q)
{
	while(q--);
}

//1602�����
void write_com(uchar com)
{
	e=0;
	rs=0;
	rw=0;
	P0=com;
	delay_uint(25);
	e=1;
	delay_uint(100);
	e=0;
}

//1602д���ݺ���
void write_data(uchar dat)
{
	e=0;
	rs=1;
	rw=0;
	P0=dat;
	delay_uint(25);
	e=1;
	delay_uint(100);
	e=0;	
}

//�ı�Һ��λֵ����
//��ʽ�����У��У���ֵ����
void write_string(uchar hang,uchar add,uchar *p)
{
	if(hang==1)   
		write_com(0x80+add);
	else
		write_com(0x80+0x40+add);
	while(1)
	{
		if(*p == '\0')  break;
		write_data(*p);
		p++;
	}	
}

//���ƹ�꺯��
void write_guanbiao(uchar hang,uchar add,uchar date)
{		
	if(hang==1)   
		write_com(0x80+add);
	else
		write_com(0x80+0x40+add);
	if(date == 1)
		write_com(0x0f);     //��ʾ��겢��˸
	else 
		write_com(0x0c);   //�رչ��
}	

//1602��ʾ��λʮ������
void write_sfm1(uchar hang,uchar add,uchar date)
{
	if(hang==1)   
		write_com(0x80+add);
	else
		write_com(0x80+0x40+add);
	write_data(table_num[date%10]);	
}

void write_sfm2_ds1302(uchar hang,uchar add,uchar date)
{
	if(hang==1)
		write_com(0x80+add);
	else
		write_com(0x80+0x40+add);	  
	write_data(table_num[date/10]);
	write_data(table_num[date%10]);	
}

void write_sfm3_18B20(uchar hang,uchar add,uint date)
{
	if(hang==1)   
		write_com(0x80+add);
	else
		write_com(0x80+0x40+add);
	write_data(0x30+date/100%10);
	write_data(0x30+date/10%10);
	write_data('.');
	write_data(0x30+date%10);	
}

void write_sfm3(uchar hang,uchar add,uint date)
{
	if(hang==1)   
		write_com(0x80+add);
	else
		write_com(0x80+0x40+add);
	write_data(0x30+date/100%10);
	write_data(0x30+date/10%10);
	write_data(0x30+date%10);	
}

//1602��ʼ������
void init_1602()
{
	write_com(0x38);
	write_com(0x0c);
	write_com(0x06);
	delay_uint(1000);
	write_string(1,0,"000/min 00:00:00");	
	write_string(2,0,"  H:000  L:000  ");
	write_sfm3(2,4,rate_h);	     //��ʾ���� ʱ
	write_sfm3(2,11,rate_l);	   //��ʾ���� ��
}

//����Ӧ��ַ��д������
void write_ds1302(uchar add,uchar dat)
{		
	rst = 1;             //��λ������
	for(i=0;i<8;i++)
	{                    //��λ��ǰ
		clk = 0;		       //ʱ�������Ϳ�ʼд������
		io = add & 0x01;    	
		add >>= 1;		     //�ѵ�ַ����һλ
		clk = 1;		       //ʱ��������
	}	
	for(i=0;i<8;i++)
	{
		clk = 0;		       //ʱ�������Ϳ�ʼд����
		io = dat & 0x01;
		dat >>= 1;    		 //�ѵ�ַ����һλ
		clk = 1;		       //ʱ��������
	}
	rst = 0;			       //��λ������
}

//�Ӷ�Ӧ��ַ�ж�������
uchar read_ds1302(uchar add)
{
	uchar value,i;
	rst = 1;			       //��λ������
	for(i=0;i<8;i++)
	{				             //��λ��ǰ
		clk = 0;	       	 //ʱ�������Ϳ�ʼд������
		io = add & 0x01;    	
		add >>= 1;		     //�ѵ�ַ����һλ
		clk = 1;		       //ʱ��������
	}		
	for(i=0;i<8;i++)
	{
		clk = 0;		       //ʱ�������Ϳ�ʼ��������
		value >>= 1;
		if(io == 1)
			value |= 0x80;
		clk = 1;		       //ʱ��������
	}
	rst = 0;			       //��λ������
	return value;		     //���ض�����ֵ
}

//����ʱ��
void read_time()
{
	miao = read_ds1302(read_add[0]);	//����
	fen  = read_ds1302(read_add[1]);	//����
	shi  = read_ds1302(read_add[2]);	//��ʱ
	ri   = read_ds1302(read_add[3]);	//����
	yue  = read_ds1302(read_add[4]);	//����
	nian = read_ds1302(read_add[5]);	//����
	week = read_ds1302(read_add[6]);	//������
}

//��ds1302��д��ʱ��
void write_time()
{
	write_ds1302(0x8e,0x00);		    	//��д����
	write_ds1302(write_add[0],miao);	//д��
	write_ds1302(write_add[1],fen);		//д��
	write_ds1302(write_add[2],shi);		//дʱ
	write_ds1302(write_add[3],ri);		//д��
	write_ds1302(write_add[4],yue);		//д��
	write_ds1302(write_add[5],nian);	//д����
	write_ds1302(write_add[6],week);	//д��
	write_ds1302(0x8e,0x80);		    	//�ر�д����
}

//18b20��ʼ������
void init_18b20()
{
	bit q;
	dq = 1;				      //��������
	delay_uint(1);	    //��ʱ15us
	dq = 0;				      //��λ����
	delay_uint(80);		  //��ʱ750us
	dq = 1;			      	//�������� �ȴ�
	delay_uint(10);		  //��ʱ110us
	q = dq;			       	//��ȡ18b20��ʼ���ź�
	delay_uint(20);		  //��ʱ200us
	dq = 1;				      //�������� �ͷ�
}

//д18b20������
void write_18b20(uchar dat)
{
	uchar i;
	for(i=0;i<8;i++)
	{
		dq = 0;		      	 //�������� ��ʼдʱ��϶
		dq = dat & 0x01;   //��18b20д����
		delay_uint(5);	   //��ʱ60us
		dq = 1;			       //�ͷ�����
		dat >>= 1;
	}	
}

//��ȡ18b20������
uchar read_18b20()
{
	uchar i,value;
	for(i=0;i<8;i++)
	{
		dq = 0;			       //�������� ��ʼ��ʱ��϶
		value >>= 1;    	 
		dq = 1;			       //�ͷ�����
		if(dq == 1)	    	 //��ʼ��д����
			value |= 0x80;
		delay_uint(5);	   //��ʱ60us
	}
	return value;		     //��������
}

//��ȡ�¶�
uint read_temp()
{
	uint value;
	uchar low;
	init_18b20();		     //��ʼ��18b20
	write_18b20(0xcc);
	write_18b20(0x44);
	delay_uint(50);	
	
	init_18b20();		     //��ʼ��18b20
	EA = 0;
	write_18b20(0xcc);	
	write_18b20(0xbe);	  
	EA = 1;

	low = read_18b20();	   //�¶ȵ��ֽ�
	value = read_18b20();  //�¶ȸ��ֽ�

	value <<= 8;		       //�¶ȸ�λ����8λ
	value |= low;		       //�����¶ȷ���value�Ͱ�λ��
	value *= 0.625;	   
	return value;		       //��������
}

//��������
uchar key_can;

void key()
{
	static uchar key_new;
	key_can = 20;         
	P3 |= 0x78;           
	if((P3 & 0x78) != 0x78)	//��������
	{
		delay_1ms(1);	      	//��ʱ����
		if(((P3 & 0x78) != 0x78) && (key_new == 1))
		{			
			key_new = 0;
			switch(P3 & 0x78)
			{
				case 0x70:  key_can = 4;  break;
				case 0x68:  key_can = 3;  break;
				case 0x58:  key_can = 2;  break;
				case 0x38:  key_can = 1;  break;
			}
		}			
	}
	else 
		key_new = 1;	
}

//��������
void key_with()
{
	if(key_can == 1)
	{
		menu_1++;
		if(menu_1 == 1) //���ð���
		{
			menu_2 = 1;
			write_string(1,0,"    -  -    W:  ");			
			write_string(2,0," 20  -  -       ");	
		}
		if(menu_1 == 2)
		{
			menu_2 = 1;
			write_string(1,0,"   set alarm    ");			
			write_string(2,0,"  H:000  L:000  ");
		}
		if(menu_1 > 2)
		{
			menu_1 = 0;
			write_guanbiao(1,2,0);
			init_1602();
		}
	}
	if(key_can == 2) //ѡ�񰴼�
	{
		if(menu_1 == 1)
		{
			menu_2 ++;
			if(menu_2 > 7)
				menu_2 = 1;
		}
		if(menu_1 == 2)
		{
			menu_2 ++;
			if(menu_2 > 2)
				menu_2 = 1;				
		}
	}
	if(menu_1 == 1)
	{
		if(menu_2 == 1)  //����ʱ
		{
			if(key_can == 3)  //��
			{
				shi+=0x01;
				if((shi & 0x0f) >= 0x0a)
					shi = (shi & 0xf0) + 0x10;
				if(shi >= 0x24)
					shi = 0;
			}		
			if(key_can == 4)  //��
			{
				if(shi == 0x00)
					shi = 0x24;
				if((shi & 0x0f) == 0x00)
					shi = (shi | 0x0a) - 0x10;
				shi -- ; 
			}	  				
		}
		if(menu_2 == 2)  //���÷�
		{
			if(key_can == 3)  //��
			{
				fen+=0x01;
				if((fen & 0x0f) >= 0x0a)
					fen = (fen & 0xf0) + 0x10;
				if(fen >= 0x60)
					fen = 0;
			}		
			if(key_can == 4)  //��
			{
				if(fen == 0x00)
					fen = 0x5a;
				if((fen & 0x0f) == 0x00)
					fen = (fen | 0x0a) - 0x10;
				fen -- ;
			}	
		}
		if(menu_2 == 3)  //������
		{
			if(key_can == 3)  //��
			{
				miao+=0x01;
				if((miao & 0x0f) >= 0x0a)
					miao = (miao & 0xf0) + 0x10;
				if(miao >= 0x60)
					miao = 0;
			}	
			if(key_can == 4)  //��
			{
				if(miao == 0x00)
					miao = 0x5a;
				if((miao & 0x0f) == 0x00)
					miao = (miao | 0x0a) - 0x10;
				miao -- ;			
			}
		}
		if(menu_2 == 4)  //��������
		{
			if(key_can == 3)  //��
			{
	    		week+=0x01;
				if((week & 0x0f) >= 0x0a)
					week = (week & 0xf0) + 0x10;
				if(week >= 0x08)
					week = 1;
			}		
			if(key_can == 4)  //��
			{
				if(week == 0x01)
					week = 0x08;
				if((week & 0x0f) == 0x00)
					week = (week | 0x0a) - 0x10;
				week -- ;
			}	
		}
		if(menu_2 == 5)  //������
		{
			if(key_can == 3)  //��
			{
		    	nian+=0x01;
				if((nian & 0x0f) >= 0x0a)
					nian = (nian & 0xf0) + 0x10;
				if(nian >= 0x9a)
					nian = 1;
			}		
			if(key_can == 4)  //��
			{
				if(nian == 0x01)
					nian = 0x9a;
				if((nian & 0x0f) == 0x00)
					nian = (nian | 0x0a) - 0x10;
				nian -- ;		
			}	
		}
		if(menu_2 == 6)  //������
		{
			if(key_can == 3)  //��
			{
		    	yue+=0x01;
				if((yue & 0x0f) >= 0x0a)
					yue = (yue & 0xf0) + 0x10;
				if(yue >= 0x13)
					yue = 1;
			}		
			if(key_can == 4)  //��
			{
				if(yue == 0x01)
					yue = 0x13;
				if((yue & 0x0f) == 0x00)
					yue = (yue | 0x0a) - 0x10;
				yue -- ;					
			}	
		}
		if(menu_2 == 7)  //������
		{
			if(key_can == 3)  //��
			{
	    	ri+=0x01;
			if((ri & 0x0f) >= 0x0a)
				ri = (ri & 0xf0) + 0x10;
			if(ri >= 0x32)
				ri = 0;			
			}		
			if(key_can == 4)  //��
			{
				if(ri == 0x01)
					ri = 0x32;
				if((ri & 0x0f) == 0x00)
					ri = (ri | 0x0a) - 0x10;
				ri -- ;			
			}	
		}
		//������ʾ
		write_sfm2_ds1302(1,2,shi);
		write_sfm2_ds1302(1,5,fen);	 
		write_sfm2_ds1302(1,8,miao);
		write_sfm1(1,14,week);	  				
		write_sfm2_ds1302(2,3,nian);
		write_sfm2_ds1302(2,6,yue);	
		write_sfm2_ds1302(2,9,ri);	
		switch(menu_2)  //���
		{
			case 1:  write_guanbiao(1,2,1);  break;
			case 2:  write_guanbiao(1,5,1);  break;
			case 3:  write_guanbiao(1,8,1);  break;
			case 4:  write_guanbiao(1,14,1);  break;
			case 5:  write_guanbiao(2,3,1);  break;
			case 6:  write_guanbiao(2,6,1);  break;
			case 7:  write_guanbiao(2,9,1);  break;
		}
		write_time();	   //д��ʱ��
	}	

//����������
	if(menu_1 == 2)
	{
		if(menu_2 == 1)  //�������ޱ���
		{
			if(key_can == 3)  //��
			{
		    	rate_h ++;
				if(rate_h >= 255)
					rate_h = 0;
			}		
			if(key_can == 4)  //��
			{
				rate_h -- ;
				if(rate_h <= rate_l)
					rate_h = rate_l + 1;
			}	
		}
		if(menu_2 == 2)  //�������ޱ���
		{
			if(key_can == 3)  //��
			{
	    		rate_l ++;
				if(rate_l >= rate_h)
					rate_l = rate_h - 1;
			}	
			if(key_can == 4)  //��
			{
				if(rate_l == 0x00)
					rate_l = 1;
				rate_l -- ;			
			}
		}
		write_sfm3(2,4,rate_h);
		write_sfm3(2,11,rate_l);
		switch(menu_2)
		{
			case 1:  write_guanbiao(2,4,1);  break;
			case 2:  write_guanbiao(2,11,1);  break;
		}	
	}	
}

//��ʱ��0��ʼ��
void time_init()	  
{
	EA   = 1;
	TMOD = 0X01;
	ET0  = 1;		
	TR0  = 1;	
}

//�ⲿ�ж�0��ʼ������
void init_int0()
{
	EX0=1;
	EA=1;	
	IT0 = 1; 
}

//��������
void clock_h_l()
{
	if((rate <= rate_l) || (rate >= rate_h))
	{
		beep = ~beep;
	}
	else
	{
		beep = 1;	
	}			
}

//������
void main()
{	
	static uint value;
	time_init();  //��ʱ����ʼ��
	init_int0();  //�ⲿ�ж�0��ʼ��
	init_1602();  //1602��ʼ��
	while(1)
	{	
		key();  //��������
		if(key_can < 10)
		{
			key_with();	
		}	
		value ++;
		if(value >= 300)
		{
			value = 0;
			if(menu_1 == 0)
			{
				if(displayOK==0)
				{
				 	rate = 0;
				}
				else
				{
					rate=60000/(time[1]/5+time[2]/5+time[3]/5+time[4]/5+time[5]/5);
				}
				write_sfm2_ds1302(1,8,shi);
				write_sfm2_ds1302(1,11,fen);
				write_sfm2_ds1302(1,14,miao);	
				temperature = read_temp();
				write_sfm3_18B20(1,10,temperature);
				read_time();
				write_sfm3(1,0,rate);
				if(rate != 0)
					clock_h_l();
				else 
					beep = 1;
			}
		}
		delay_1ms(1);
	}
}


void int0() interrupt 0
{
	if(timecount<8)
	{
			TR0=1;
	}
	else
	{
		time[i]=timecount*50+TH0+TL0;
		timecount=0;
		i++;
		if(i==6)
		{
			i=1;
			displayOK=1; 
		}								
	}
}

//��ʱ��0�жϳ���
void time0_int() interrupt 1
{	
	TH0 = 0x3c;
	TL0 = 0xb0; 
	timecount++;
	if(timecount>25)
	{
			i=0;
			timecount=0;
			displayOK=0;
			TR0=0;
	}
	if(menu_1 > 2)
	{
			menu_1 = 0;
			write_guanbiao(1,2,0);
			init_1602();  	
	}
	if(key_can == 2)
	{
		if(menu_1 == 1)		
		{
			menu_2 ++;
			if(menu_2 > 7)
				menu_2 = 1;
		}
		if(menu_1 == 2)		
		{
			menu_2 ++;
			if(menu_2 > 2)
				menu_2 = 1;				
		}
	}
	if(menu_1 == 1)
	{
		if(menu_2 == 1)	
		{
			if(key_can == 3)
			{
				shi+=0x01;
				if((shi & 0x0f) >= 0x0a)
					shi = (shi & 0xf0) + 0x10;
				if(shi >= 0x24)
					shi = 0;
			}		
			if(key_can == 4)
			{
				if(shi == 0x00)
					shi = 0x24;
				if((shi & 0x0f) == 0x00)
					shi = (shi | 0x0a) - 0x10;
				shi -- ; 
			}	  				
		}
		if(menu_2 == 2)		
		{
			if(key_can == 3)	
			{
				fen+=0x01;
				if((fen & 0x0f) >= 0x0a)
					fen = (fen & 0xf0) + 0x10;
				if(fen >= 0x60)
					fen = 0;
			}		
			if(key_can == 4)
			{
				if(fen == 0x00)
					fen = 0x5a;
				if((fen & 0x0f) == 0x00)
					fen = (fen | 0x0a) - 0x10;
				fen -- ;
			}	
		}
		if(menu_2 == 3)		
		{
			if(key_can == 3)	
			{
				miao+=0x01;
				if((miao & 0x0f) >= 0x0a)
					miao = (miao & 0xf0) + 0x10;
				if(miao >= 0x60)
					miao = 0;
			}	
			if(key_can == 4)	  
			{
				if(miao == 0x00)
					miao = 0x5a;
				if((miao & 0x0f) == 0x00)
					miao = (miao | 0x0a) - 0x10;
				miao -- ;			
			}
		}
		if(menu_2 == 4)	
		{
			if(key_can == 3)
			{
	    		week+=0x01;
				if((week & 0x0f) >= 0x0a)
					week = (week & 0xf0) + 0x10;
				if(week >= 0x08)
					week = 1;
			}		
			if(key_can == 4)
			{
				if(week == 0x01)
					week = 0x08;
				if((week & 0x0f) == 0x00)
					week = (week | 0x0a) - 0x10;
				week -- ;
			}	
		}
		if(menu_2 == 5)	
		{
			if(key_can == 3)
			{
		    	nian+=0x01;
				if((nian & 0x0f) >= 0x0a)
					nian = (nian & 0xf0) + 0x10;
				if(nian >= 0x9a)
					nian = 1;
			}		
			if(key_can == 4)
			{
				if(nian == 0x01)
					nian = 0x9a;
				if((nian & 0x0f) == 0x00)
					nian = (nian | 0x0a) - 0x10;
				nian -- ;		
			}	
		}
		if(menu_2 == 6)
		{
			if(key_can == 3)
			{
		    	yue+=0x01;
				if((yue & 0x0f) >= 0x0a)
					yue = (yue & 0xf0) + 0x10;
				if(yue >= 0x13)
					yue = 1;
			}		
			if(key_can == 4)
			{
				if(yue == 0x01)
					yue = 0x13;
				if((yue & 0x0f) == 0x00)
					yue = (yue | 0x0a) - 0x10;
				yue -- ;					
			}	
		}
		if(menu_2 == 7)	
		{
			if(key_can == 3)
			{
	    	ri+=0x01;
			if((ri & 0x0f) >= 0x0a)
				ri = (ri & 0xf0) + 0x10;
			if(ri >= 0x32)
				ri = 0;			
			}		
			if(key_can == 4)
			{
				if(ri == 0x01)
					ri = 0x32;
				if((ri & 0x0f) == 0x00)
					ri = (ri | 0x0a) - 0x10;
				ri -- ;			
			}	
		}
		write_sfm2_ds1302(1,2,shi);	
		write_sfm2_ds1302(1,5,fen);	 
		write_sfm2_ds1302(1,8,miao);	
		write_sfm1(1,14,week);
		write_sfm2_ds1302(2,3,nian);	
		write_sfm2_ds1302(2,6,yue);
		write_sfm2_ds1302(2,9,ri);
		switch(menu_2)  //�����ʾ
		{
			case 1:  write_guanbiao(1,2,1);  break;
			case 2:  write_guanbiao(1,5,1);  break;
			case 3:  write_guanbiao(1,8,1);  break;
			case 4:  write_guanbiao(1,14,1);  break;
			case 5:  write_guanbiao(2,3,1);  break;
			case 6:  write_guanbiao(2,6,1);  break;
			case 7:  write_guanbiao(2,9,1);  break;
		}
		write_time();  //д��ʱ��
	}	

//����
	if(menu_1 == 2)
	{
		if(menu_2 == 1)
		{
			if(key_can == 3)
			{
		    	rate_h ++;
				if(rate_h >= 255)
					rate_h = 0;
			}		
			if(key_can == 4)
			{
				rate_h -- ;
				if(rate_h <= rate_l)
					rate_h = rate_l + 1;
			}	
		}
		if(menu_2 == 2)
		{
			if(key_can == 3)
			{
	    		rate_l ++;
				if(rate_l >= rate_h)
					rate_l = rate_h - 1;
			}	
			if(key_can == 4)
			{
				if(rate_l == 0x00)
					rate_l = 1;
				rate_l -- ;			
			}
		}
		write_sfm3(2,4,rate_h);	
		write_sfm3(2,11,rate_l);	
		switch(menu_2)	
		{
			case 1:  write_guanbiao(2,4,1);  break;
			case 2:  write_guanbiao(2,11,1);  break;
		}	
	}	
}