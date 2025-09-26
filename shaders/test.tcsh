#version 450

// 关键布局声明：定义每个Patch输出3个控制点（用于三角形细分）
layout(vertices = 3) out;

// 输入来自顶点着色器的数据
layout(location = 0) in vec2 vTexCoord[];
layout(location = 1) in vec3 vNormal[];

// 输出到细分计算着色器(TES)的数据
layout(location = 0) out vec2 tcTexCoord[];
layout(location = 1) out vec3 tcNormal[];

void main() {
	// 设置细分级别
	// 内层细分级别：影响三角形内部的细分
	gl_TessLevelInner[0] = 3.0;

	// 外层细分级别：影响三角形每条边的细分
	gl_TessLevelOuter[0] = 2.0;
	gl_TessLevelOuter[1] = 3.0;
	gl_TessLevelOuter[2] = 4.0;

	// 传递控制点数据
	// gl_InvocationID 标识当前正在处理的输出控制点(0, 1, 2)
	gl_out[gl_InvocationID].gl_Position = gl_in[gl_InvocationID].gl_Position;

	// 传递自定义属性
	tcTexCoord[gl_InvocationID] = vTexCoord[gl_InvocationID];
	tcNormal[gl_InvocationID] = vNormal[gl_InvocationID];
}
