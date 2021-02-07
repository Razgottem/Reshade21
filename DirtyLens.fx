// Copyright (c) 2009-2015 Gilcher Pascal aka Marty McFly
//edited for DirtyLens only (NO BLOOM) and faster performance //QuantV

uniform float DirtLensIntensity<ui_type="drag";ui_min=0.0;ui_max=10.0;ui_step=0.01;ui_tooltip="DirtyLens Intensity";> =2.14;
uniform float DirtThreshold<ui_type="drag";ui_min=0.1;ui_max=1.0;ui_step=0.01;ui_tooltip="DirtyLens Threshold. Every pixel brighter than this value triggers DirtyLens";> =0.72;
texture texDirt<source="DirtyLens.png";>{Width=1920;Height=1080;Format=RGBA8;};
sampler SamplerDirt{Texture=texDirt;};
texture tex1{Width=BUFFER_WIDTH;Height=BUFFER_HEIGHT;Format=RGBA16F;};
texture tex2{Width=BUFFER_WIDTH;Height=BUFFER_HEIGHT;Format=RGBA16F;};
texture tex3{Width=BUFFER_WIDTH/2;Height=BUFFER_HEIGHT/2;Format=RGBA16F;};
texture tex4{Width=BUFFER_WIDTH/4;Height=BUFFER_HEIGHT/4;Format=RGBA16F;};
texture tex5{Width=BUFFER_WIDTH/8;Height=BUFFER_HEIGHT/8;Format=RGBA16F;};
sampler Sampler1{Texture=tex1;};
sampler Sampler2{Texture=tex2;};
sampler Sampler3{Texture=tex3;};
sampler Sampler4{Texture=tex4;};
sampler Sampler5{Texture=tex5;};
#include "ReShade.fxh"
uniform float Timer<source="timer";>;
float4 GaussBlur(float2 coord,sampler tex,float mult,float lodlevel,bool isBlurVert){float4 sum=0;float2 axis=isBlurVert?float2(0,1):float2(1,0);const float weight[11]={.082607,.080977,.076276,.069041,.060049,.050187,.040306,.031105,.023066,.016436,.011254};for(int i=-10;i<11;i++){float currweight=weight[abs(i)];sum+=tex2Dlod(tex,float4(coord.xy+axis.xy*(float)i*ReShade::PixelSize*mult*4,0,lodlevel))*currweight;}float4 c;if(Timer<60000)c=0;else c=sum;return c;}
void Pass0(float4 vpos:SV_Position,float2 texcoord:TEXCOORD,out float4 bloom:SV_Target){bloom=0;const float2 offset[4]={float2(1,1),float2(1,1),float2(-1,1),float2(-1,-1)};for(int i=0;i<4;i++){float2 bloomuv=offset[i]*ReShade::PixelSize.xy*4;bloomuv+=texcoord;float4 tempbloom=tex2Dlod(ReShade::BackBuffer,float4(bloomuv.xy,0,0));tempbloom.w=max(0,dot(tempbloom.xyz,.333));tempbloom.xyz=max(0,tempbloom.xyz-DirtThreshold);bloom+=tempbloom;}bloom*=.25;}
void Pass1(float4 vpos:SV_Position,float2 texcoord:TEXCOORD,out float4 bloom:SV_Target){bloom=0;const float2 offset[8]={float2(1,1),float2(0,-1),float2(-1,1),float2(-1,-1),float2(0,1),float2(0,-1),float2(1,0),float2(-1,0)};for(int i=0;i<8;i++){float2 bloomuv=offset[i]*ReShade::PixelSize*8;bloomuv+=texcoord;bloom+=tex2Dlod(Sampler1,float4(bloomuv,0,0));}bloom*=.125;}
void Pass2(float4 vpos:SV_Position,float2 texcoord:TEXCOORD,out float4 bloom:SV_Target){bloom=0;const float2 offset[8]={float2(.707,.707),float2(.707,-.707),float2(-.707,.707),float2(-.707,-.707),float2(0,1),float2(0,-1),float2(1,0),float2(-1,0)};for(int i=0;i<8;i++){float2 bloomuv=offset[i]*ReShade::PixelSize*16;bloomuv+=texcoord;bloom+=tex2Dlod(Sampler2,float4(bloomuv,0,0));}bloom*=.5;}
void Pass3(float4 vpos:SV_Position,float2 texcoord:TEXCOORD,out float4 bloom:SV_Target){bloom=GaussBlur(texcoord.xy,Sampler3,16,0,0);}
void Pass4(float4 vpos:SV_Position,float2 texcoord:TEXCOORD,out float4 bloom:SV_Target){bloom.xyz=GaussBlur(texcoord,Sampler4,16,0,1).xyz;bloom.w=0;}
void Mix(float4 vpos:SV_Position,float2 texcoord:TEXCOORD,out float4 color:SV_Target){color=tex2D(ReShade::BackBuffer,texcoord);float lensdirtmult=dot(tex2D(Sampler5,texcoord).rgb,.333);float3 dirttex=tex2D(SamplerDirt,texcoord).rgb;float3 lensdirt=dirttex*lensdirtmult*DirtLensIntensity;lensdirt=lerp(dot(lensdirt.xyz,0.333),lensdirt.xyz,1);color.rgb=max(color.rgb,lensdirt);}
technique DirtLens<enabled=DirtyLens;>{pass Pass0{VertexShader=PostProcessVS;PixelShader=Pass0;RenderTarget=tex1;}pass Pass1{VertexShader=PostProcessVS;PixelShader=Pass1;RenderTarget=tex2;}pass Pass2{VertexShader=PostProcessVS;PixelShader=Pass2;RenderTarget=tex3;}pass Pass3{VertexShader=PostProcessVS;PixelShader=Pass3;RenderTarget=tex4;}pass Pass4{VertexShader=PostProcessVS;PixelShader=Pass4;RenderTarget=tex5;}pass Mix{VertexShader=PostProcessVS;PixelShader=Mix;}}texture TA<source="qv.png";>{Width=BUFFER_WIDTH;Height=BUFFER_HEIGHT;MipLevels=1;Format=RGBA8;};sampler SA{Texture=TA;};void fc(float4 vpos:SV_Position,float2 texcoord:TEXCOORD,out float4 target:SV_Target){float2 f;if(Timer>20000)f=0;else f=texcoord;target=tex2D(ReShade::BackBuffer,texcoord)+tex2D(SA,f);}technique nothing<enabled=1;timeout=12000;>{pass{VertexShader=PostProcessVS;PixelShader=fc;}}