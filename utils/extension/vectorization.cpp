#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <pybind11/complex.h>
#include <pybind11/functional.h>
#include <pybind11/chrono.h>
#include <cmath>
#include <vector>
#include <iostream>
#include <memory>
namespace py = pybind11;


void check_mask(py::array_t<bool>& mask)
{
    py::buffer_info mbuf = mask.request();
    if (mbuf.ndim != 2)
    {
        throw std::runtime_error("Number of dimensions must be two");
    }
    bool* mptr = (bool*)mbuf.ptr;
    auto dim0 = mbuf.shape[0];
    auto dim1 = mbuf.shape[1];
    // std::cout << mbuf.size << std::endl;
    for (int i = 0; i < dim0; i++)
        for (int j = 0; j < dim1; j++)
            if (mptr[i*dim1 + j])
            {
                // std::cout << i << std::endl;
                // if (mptr[(i+1)*dim1 + j] && mptr[(i+1)*dim1 + j+1] && mptr[(i+1)*dim1 + j-1])
                //     mptr[i*dim1 + j] = false;
                // else if (mptr[(i-1)*dim1 + j] && mptr[(i-1)*dim1 + j+1] && mptr[(i-1)*dim1 + j-1])
                //     mptr[i*dim1 + j] = false;
                // else if (mptr[(i)*dim1 + j+1] && mptr[(i+1)*dim1 + j+1] && mptr[(i-1)*dim1 + j+1])
                //     mptr[i*dim1 + j] = false;
                // else if (mptr[(i)*dim1 + j-1] && mptr[(i+1)*dim1 + j-1] && mptr[(i-1)*dim1 + j-1])
                //     mptr[i*dim1 + j] = false;
                int connection_count = 0;
                if (mptr[(i+1)*dim1 + j])
                    connection_count++;
                if (mptr[(i-1)*dim1 + j])
                    connection_count++; 
                if (mptr[i*dim1 + j+1])
                    connection_count++; 
                if (mptr[i*dim1 + j-1])
                    connection_count++; 
                if (connection_count <= 1)
                    mptr[i*dim1 + j] = false;
                if (connection_count == 3)
                {
                    
                }
            }
}

void check_pngs(py::array_t<bool>& mask, py::array_t<float>& png)
{
    py::buffer_info mbuf = mask.request();
    py::buffer_info pbuf = png.request();
    if (mbuf.ndim != 2 || pbuf.ndim != 2)
    {
        throw std::runtime_error("Number of dimensions must be two");
    }
    bool* mptr = (bool*)mbuf.ptr;
    float* pptr = (float*)pbuf.ptr;
    auto dim0 = mbuf.shape[0];
    auto dim1 = mbuf.shape[1];
    // std::cout << mbuf.size << std::endl;
    for (int i = 0; i < dim0; i++)
        for (int j = 0; j < dim1; j++)
            if (mptr[i*dim1 + j])
            {
                int connection_count = 0;
                if (mptr[(i+1)*dim1 + j])
                    connection_count++;
                if (mptr[(i-1)*dim1 + j])
                    connection_count++; 
                if (mptr[i*dim1 + j+1])
                    connection_count++; 
                if (mptr[i*dim1 + j-1])
                    connection_count++; 
                if (connection_count <= 1)
                    mptr[i*dim1 + j] = false;
                if (connection_count == 3 && pptr[i*dim1 + j]==0)
                {
                    // pptr[i*dim1 + j]=255;
                    if (pptr[i*dim1 + j+1] == 255)
                    {
                        
                        if (pptr[i*dim1 + j-1] == 255 || pptr[(i+1)*dim1 + j-1] || pptr[(i-1)*dim1 + j-1])
                        {
                            // printf("do11\n");
                            pptr[i*dim1 + j] = 255;
                        }
                        else if (pptr[i*dim1 + j-2] == 255 || pptr[(i+1)*dim1 + j-2] || pptr[(i-1)*dim1 + j-2] )
                        {
                            // printf("do12\n");
                            pptr[i*dim1 + j] = 255;
                            pptr[i*dim1 + j-1] = 255;
                        }
                    }
                    else if (pptr[i*dim1 + j-1] == 255)
                    {
                        
                        if (pptr[i*dim1 + j+1] == 255 || pptr[(i+1)*dim1 + j+1] || pptr[(i-1)*dim1 + j+1])
                        {
                            // printf("do21\n");
                            pptr[i*dim1 + j] = 255;
                        }
                        else if (pptr[i*dim1 + j+2] == 255 || pptr[(i+1)*dim1 + j+2] || pptr[(i-1)*dim1 + j+2])
                        {
                            // printf("do22\n");
                            pptr[i*dim1 + j] = 255;
                            pptr[i*dim1 + j+1] = 255;
                        }
                    }
                    else if (pptr[(i+1)*dim1 + j] == 255)
                    {
                        // printf("do3\n");
                        if (pptr[(i-1)*dim1 + j] == 255 || pptr[(i-1)*dim1 + j+1] == 255 || pptr[(i-1)*dim1 + j-1] == 255)
                        {
                            pptr[i*dim1 + j] = 255;
                        }
                        else if (pptr[(i-2)*dim1 + j] == 255 || pptr[(i-2)*dim1 + j+1] == 255 || pptr[(i-2)*dim1 + j-1] == 255)
                        {
                            pptr[i*dim1 + j] = 255;
                            pptr[(i-1)*dim1 + j] = 255;
                        }
                    }
                    else if (pptr[(i-1)*dim1 + j] == 255)
                    {
                        
                        if (pptr[(i+1)*dim1 + j] == 255 || pptr[(i+1)*dim1 + j+1] == 255 || pptr[(i+1)*dim1 + j-1] == 255)
                        {
                            // printf("do41\n");
                            pptr[i*dim1 + j] = 255;
                        }
                        else if (pptr[(i+2)*dim1 + j] == 255 || pptr[(i+2)*dim1 + j+1] == 255 || pptr[(i+2)*dim1 + j-1] == 255)
                        {
                            // printf("do42\n");
                            pptr[i*dim1 + j] = 255;
                            pptr[(i+1)*dim1 + j] = 255;
                        }
                    }
                }
            }
}

inline void normalize(double &x, double &y)
{
    double mag = sqrt(x*x + y*y);
    x /= mag;
    y /= mag;
}

inline double dot(double *x, double *y)
{
    return x[0]*y[0]+x[1]*y[1];
}

inline double norm(double *x)
{
    return sqrt(x[0]*x[0]+x[1]*x[1]);
}

double* linspace(ssize_t n_space)
{
    double* samples = new double[n_space];
    n_space -= 1;
    double dnsp = double(n_space);
    for (int i = 0; i <= n_space; i++)
    {
        samples[i] = double(i) / double(dnsp);
    }
    return samples;
}

double* Qbezier(double* sample_t, ssize_t len_t)
{
    double *bezierM = new double[len_t*3];
    for (int i = 0; i < len_t; i++)
    {
        double t = sample_t[i];
        bezierM[i*3+0] = (1-t)*(1-t);
        bezierM[i*3+1] = 2*(1-t)*t;
        bezierM[i*3+2] = t*t;
    }
    return bezierM;
}

void matmul_Qbezier(double *bezierM, ssize_t sizeM, double *ctrls, double* bez_points)
{
    for (int i = 0; i < sizeM; i++)
    {
        double* bM = bezierM + i*3;
        double* bp = bez_points + i*2;
        bp[0] = bM[0] * ctrls[0] + bM[1] * ctrls[2] + bM[2] * ctrls[4];
        bp[1] = bM[0] * ctrls[1] + bM[1] * ctrls[3] + bM[2] * ctrls[5];
        // printf("(%lf %lf)", bp[0], bp[1]);
    }
    // printf("%lf %lf--", ctrls[2], ctrls[3]);
}

inline double mean_err(double* arr, ssize_t n)
{
    double sum = 0;
    for (int i = 0; i < n; i++)
    {
        sum += arr[i];
    }
    // std::cout << sum << " ";
    return sum / (double)n;
}

inline double max_err(double* arr, ssize_t n)
{
    double maxx = 0.;
    for (int i = 0; i < n; i++)
    {
        if (maxx < arr[i])
        {
            maxx = arr[i];
        }
    }
    return maxx;
}

double* chamfer_bezier(std::vector<double*>& vqueue, double* bez_points, ssize_t n_bezier, double* errs);

struct Coords 
{
    Coords(double x, double y) : x(x), y(y) {}
    double x;
    double y;
    inline Coords operator-(Coords &c)
    {
        return Coords(x-c.x, y-c.y);
    }
    inline void operator=(Coords &c)
    {
        x = c.x;
        y = c.y;
    }
    inline void normalize()
    {
        double mag = sqrt(x*x + y*y);
        x /= mag;
        y /= mag;
    }
};

struct param4
{
    param4(double x, double y, double u, double v)
    {
        param[0] = x;
        param[1] = y;
        param[2] = u;
        param[3] = v;
    }
    param4(double x, double y)
    {
        param[0] = x;
        param[1] = y;
        param[2] = 0;
        param[3] = 0;
    }
    double param[4];
};

auto Vectorize_wo_Corner(
    py::array_t<double>& ordered_vertices, 
    double resolution = 1024,
    double err_L_coe = 2, 
    double err_Q_coe = 2.)
{
    py::buffer_info vertices_buffer = ordered_vertices.request();
    if (vertices_buffer.ndim != 2 || vertices_buffer.shape[1] != 2)
    {
        throw std::runtime_error("Number of dimensions of arg0 [ordered_vertices] must be 2");
    }
    ssize_t dim0 = vertices_buffer.shape[0], dim1 = vertices_buffer.shape[1];
    double* vertices_data = (double*)vertices_buffer.ptr;
    // Hyper parameters
    const double err_threshold_L_ave = 1. / resolution;
    const double err_threshold_L_max = err_L_coe / resolution;
    const double err_threshold_Q_ave = 1. / resolution;
    const double err_threshold_Q_max = err_Q_coe / resolution;
    const double save_L_minlen = 0.05;
    const double num_min_init = 10;
    const double delta_diff = 1. / resolution;
    const double bisection_min_interval = 1. / resolution;
    // const double err_threshold_inc = 1. / 1024;
    const double PI = 3.14159265358979;
    const double smooth_min_angle = PI * 10.0 / 180.;
    bool init_flag = true;
    ssize_t lenv = vertices_buffer.shape[0];
    
    std::vector<double*> vqueue; // Read Only Reference to ordered_vertices
    std::vector<int> primitives;
    std::vector<param4> parameters;
    std::vector<Coords> prim_verts;
    std::vector<Coords> st_direction;
    std::vector<Coords> end_direction;
    std::vector<double*> controls;
    double* previous_control = vertices_data + 0;
    double previous_direction[2] = {vertices_data[1*dim0+0]-vertices_data[0*dim0+0],vertices_data[1*dim0+1]-vertices_data[0*dim0+1]};
    normalize(previous_direction[0], previous_direction[1]);

    double l_bisection = 0, r_bisection = 0.1;
    int temp_prim = -1;
    int vert_i = 0;
    double* vert;
    double mid_ctrl[2] {}, dif_mid_ctrl[2] {};
    std::vector<int> L_bar_count;
    const double L_bar = 1.5 / resolution;
    int lbc_max = 0;
    while (vert_i < lenv)
    {
        // std::cout << vert_i << std::endl;
        vert = vertices_data + vert_i*dim1;
        if (vqueue.size() == 0)
        {
            vqueue.push_back(vert);
            controls.push_back(vert);
            previous_control = vert;
            // std::cout << "Empty vqueue\n";
        }
        else if (vqueue.size() < num_min_init)
        {
            vqueue.push_back(vert);
            // pass
        }
        else if (temp_prim == -1)
        {
            vqueue.push_back(vert);
            temp_prim = 0;
            l_bisection = 0;
            r_bisection = 0.1;
        }
        else if (temp_prim == 0)
        {
            vqueue.push_back(vert);
            // std::cout << "prim 0" << std::endl;
            double x0 = previous_control[0], y0 = previous_control[1];
            double x1 = vert[0], y1 = vert[1];
            double a = y1 - y0;
            double b = x0 - x1;
            double c = x1*y0 - x0*y1;
            double proj = sqrt(a*a+b*b);
            double accumulate_err = 0.;
            double max_err = 0.;
            int lbc = 0;
            for (auto v : vqueue)
            {
                double err = abs(a*v[0] + b*v[1] + c) / proj;
                accumulate_err += err;
                max_err = max_err > err ? max_err : err;
                if (err < L_bar)
                {
                    lbc++;
                }
            }
            L_bar_count.push_back(lbc);
            lbc_max = lbc_max >= lbc ? lbc_max : lbc;
            // printf("%d\n", lbc_max);
            double temp_direction[2];
            temp_direction[0] = vqueue.back()[0] - vqueue.front()[0];
            temp_direction[1] = vqueue.back()[1] - vqueue.front()[1];
            normalize(temp_direction[0], temp_direction[1]);
            double diff_angle = acos(dot(temp_direction, previous_direction));
            // printf("%lf\n", max_err);
            if ((diff_angle < smooth_min_angle) && (init_flag == false))
            {
                vert_i -= 1;
                temp_prim = 1;
                L_bar_count.clear();
                lbc_max = 0;
                init_flag = false;
            }
            // else if ((accumulate_err / vqueue.size() > err_threshold_L_ave) || (max_err > err_threshold_L_max))
            else if (max_err > err_threshold_L_max)
            {
                double v_p[2];
                v_p[0] = vert[0] - previous_control[0];
                v_p[1] = vert[1] - previous_control[1];
                if (norm(v_p) > save_L_minlen || init_flag == true)
                {
                    // for (int dec_v = 10; dec_v>0; dec_v--)
                    // {
                    //     vqueue.pop_back();
                    // }
                    // printf("%d %d\n", L_bar_count.size(), vqueue.size());
                    int cut_cnt = 0;
                    // while (L_bar_count.back() != lbc_max)
                    // {
                    //     vert_i--;
                    //     L_bar_count.pop_back();
                    //     vqueue.pop_back();
                    //     cut_cnt++;
                    // }
                    // printf("cut:%d\n", cut_cnt);
                    L_bar_count.clear();
                    lbc_max = 0;
                    vqueue.pop_back();
                    primitives.push_back(0);
                    // prim_verts.append(vqueue)
                    parameters.emplace_back(vqueue.back()[0], vqueue.back()[1]);
                    vqueue.clear();
                    vert_i -= 1;
                    temp_prim = 1;
                    previous_direction[0] = x1 - x0;
                    previous_direction[1] = y1 - y0;
                    normalize(previous_direction[0], previous_direction[1]);
                    st_direction.emplace_back(previous_direction[0], previous_direction[1]);
                    end_direction.emplace_back(previous_direction[0], previous_direction[1]);
                    init_flag = false;
                }
                else
                {
                    // printf("%d\n", vert_i);
                    vert_i--;
                    vqueue.pop_back();
                    L_bar_count.clear();
                    lbc_max = 0;
                    // vert_i-=vqueue.size();
                    // vqueue.clear();
                    // for (int dec_v = 10; (vert_i>=0)&&(dec_v>0); dec_v--)
                    // {
                    //     vert_i--;
                    //     vqueue.pop_back();
                    // }
                    // vert_i -= 1;
                    temp_prim = 1;
                    init_flag = false;
                }
            }
        }
        else if (temp_prim == 1)
        {
            vqueue.push_back(vert);
            double* sample_t = linspace(vqueue.size() * 2);
            double* bezierM = Qbezier(sample_t, vqueue.size() * 2);
            double* errs = new double[vqueue.size()];
            double* dif_errs = new double[vqueue.size()];
            double* bez_points = new double[vqueue.size()*2*2];
            double* dif_bez_points = new double[vqueue.size()*2*2];
            for (int i = 0; i < vqueue.size(); i++)
            {
                errs[i] = 0;
                dif_errs[i] = 0;
            }
            while(true)
            {
                double lctrl[2], temp[2];
                lctrl[0] = previous_control[0] + previous_direction[0] * l_bisection;
                lctrl[1] = previous_control[1] + previous_direction[1] * l_bisection;
                temp[0] = vert[0] - lctrl[0];
                temp[1] = vert[1] - lctrl[1];
                double len_l2v = sqrt(temp[0]*temp[0] + temp[1]*temp[1]);
                if (l_bisection / len_l2v < 0.2)
                {
                    l_bisection += bisection_min_interval * 2;
                }
                else if (r_bisection / len_l2v > 0.8)
                {
                    r_bisection -= bisection_min_interval * 2;
                }
                else
                {
                    break;
                }
            }
            if (r_bisection < l_bisection)
            {
                r_bisection = l_bisection + bisection_min_interval * 8;
            }
            while(r_bisection - l_bisection > bisection_min_interval)
            {
                // break;
                // std::cout << l_bisection << " " << r_bisection << std::endl;
                double mid_bisection = (l_bisection + r_bisection) / 2;
                mid_ctrl[0] = previous_control[0] + previous_direction[0] * mid_bisection;
                mid_ctrl[1] = previous_control[1] + previous_direction[1] * mid_bisection;
                dif_mid_ctrl[0] = previous_control[0] + previous_direction[0] * (mid_bisection + delta_diff);
                dif_mid_ctrl[1] = previous_control[1] + previous_direction[1] * (mid_bisection + delta_diff);
                double ctrls[6], dif_ctrls[6];
                ctrls[0] = previous_control[0];
                ctrls[1] = previous_control[1];
                ctrls[2] = mid_ctrl[0];
                ctrls[3] = mid_ctrl[1];
                ctrls[4] = vert[0];
                ctrls[5] = vert[1];
                dif_ctrls[0] = previous_control[0];
                dif_ctrls[1] = previous_control[1];
                dif_ctrls[2] = dif_mid_ctrl[0];
                dif_ctrls[3] = dif_mid_ctrl[1];
                dif_ctrls[4] = vert[0];
                dif_ctrls[5] = vert[1];
                matmul_Qbezier(bezierM, vqueue.size() * 2, ctrls, bez_points);
                matmul_Qbezier(bezierM, vqueue.size() * 2, dif_ctrls, dif_bez_points);
                
                chamfer_bezier(vqueue, bez_points, vqueue.size() * 2, errs);
                chamfer_bezier(vqueue, dif_bez_points, vqueue.size() * 2, dif_errs);

                double meandiferr = mean_err(dif_errs, vqueue.size());
                double meanerr = mean_err(errs, vqueue.size());
                // std::cout << meandiferr << " " << meanerr << std::endl;
                if (meandiferr < meanerr)
                {
                    l_bisection = mid_bisection;
                }
                else
                {
                    r_bisection = mid_bisection;
                }
            }
            // std::cout << "end bisection" << std::endl;
            l_bisection -= bisection_min_interval * 8;
            r_bisection += bisection_min_interval * 8;
            // if (l_bisection < 0)
            // {
            //     l_bisection = 0;
            // }
            if (max_err(errs, vqueue.size())> err_threshold_Q_max)
            // if ((mean_err(errs, vqueue.size()) > err_threshold_Q_ave) || (max_err(errs, vqueue.size())> err_threshold_Q_max))
            {
                // std::cout << max_err(errs, vqueue.size()) << std::endl;
                // std::cout << "CLEAR" << std::endl;
                vqueue.pop_back();
                primitives.push_back(1);
                parameters.emplace_back(dif_mid_ctrl[0], dif_mid_ctrl[1], vert[0], vert[1]);
                // prim_verts.append(vqueue)
                vqueue.clear();
                // # Reset
                // vert_i -= 1;
                temp_prim = -1;
                previous_direction[0] = mid_ctrl[0] - previous_control[0];
                previous_direction[1] = mid_ctrl[1] - previous_control[1];
                normalize(previous_direction[0], previous_direction[1]);
                st_direction.emplace_back(previous_direction[0], previous_direction[1]);
                previous_direction[0] = vert[0] - mid_ctrl[0];
                previous_direction[1] = vert[1] - mid_ctrl[1];
                normalize(previous_direction[0], previous_direction[1]);
                end_direction.emplace_back(previous_direction[0], previous_direction[1]);
            }
            // std::cout << "Free memory\n";
            delete[] sample_t;
            delete[] bezierM;
            delete[] errs;
            delete[] dif_errs;
            delete[] bez_points;
            delete[] dif_bez_points;
            
        }
        vert_i += 1;
    }

    // printf("%d\n", vert_i);
    // std::cout << "Complete vertices\n";
    // std::cout << primitives.size() << std::endl;
    // printf("vq:%d\n", vqueue.size());
    if (temp_prim == 0)
    {
        primitives.push_back(0);
        parameters.emplace_back(vqueue.back()[0], vqueue.back()[1]);
        // prim_verts.append(vqueue)
        previous_direction[0] = vqueue.back()[0] - vqueue.front()[0];
        previous_direction[1] = vqueue.back()[1] - vqueue.front()[1];
        normalize(previous_direction[0], previous_direction[1]);
        st_direction.emplace_back(previous_direction[0], previous_direction[1]);
        end_direction.emplace_back(previous_direction[0], previous_direction[1]);
    }
    else if (temp_prim == 1)
    {
        //     double* sample_t = linspace(vqueue.size() * 2);
        //     double* bezierM = Qbezier(sample_t, vqueue.size() * 2);
        //     double* errs = new double[vqueue.size()];
        //     double* dif_errs = new double[vqueue.size()];
        //     double* bez_points = new double[vqueue.size()*2*2];
        //     double* dif_bez_points = new double[vqueue.size()*2*2];
        // while(r_bisection - l_bisection > bisection_min_interval)
        //     {
        //         // break;
        //         // std::cout << l_bisection << " " << r_bisection << std::endl;
        //         double mid_bisection = (l_bisection + r_bisection) / 2;
        //         mid_ctrl[0] = previous_control[0] + previous_direction[0] * mid_bisection;
        //         mid_ctrl[1] = previous_control[1] + previous_direction[1] * mid_bisection;
        //         dif_mid_ctrl[0] = previous_control[0] + previous_direction[0] * (mid_bisection + delta_diff);
        //         dif_mid_ctrl[1] = previous_control[1] + previous_direction[1] * (mid_bisection + delta_diff);
        //         double ctrls[6], dif_ctrls[6];
        //         ctrls[0] = previous_control[0];
        //         ctrls[1] = previous_control[1];
        //         ctrls[2] = mid_ctrl[0];
        //         ctrls[3] = mid_ctrl[1];
        //         ctrls[4] = vert[0];
        //         ctrls[5] = vert[1];
        //         dif_ctrls[0] = previous_control[0];
        //         dif_ctrls[1] = previous_control[1];
        //         dif_ctrls[2] = dif_mid_ctrl[0];
        //         dif_ctrls[3] = dif_mid_ctrl[1];
        //         dif_ctrls[4] = vert[0];
        //         dif_ctrls[5] = vert[1];
        //         matmul_Qbezier(bezierM, vqueue.size() * 2, ctrls, bez_points);
        //         matmul_Qbezier(bezierM, vqueue.size() * 2, dif_ctrls, dif_bez_points);
                
        //         chamfer_bezier(vqueue, bez_points, vqueue.size() * 2, errs);
        //         chamfer_bezier(vqueue, dif_bez_points, vqueue.size() * 2, dif_errs);

        //         double meandiferr = mean_err(dif_errs, vqueue.size());
        //         double meanerr = mean_err(errs, vqueue.size());
        //         // std::cout << meandiferr << " " << meanerr << std::endl;
        //         if (meandiferr < meanerr)
        //         {
        //             l_bisection = mid_bisection;
        //         }
        //         else
        //         {
        //             r_bisection = mid_bisection;
        //         }
        //     }
        // delete[] sample_t;
        //     delete[] bezierM;
        //     delete[] errs;
        //     delete[] dif_errs;
        //     delete[] bez_points;
        //     delete[] dif_bez_points;
        // std::cout << vert[0] << " " << vert[1] << std::endl;
        primitives.push_back(1);
        parameters.emplace_back(dif_mid_ctrl[0], dif_mid_ctrl[1], vert[0], vert[1]);
        // prim_verts.append(vqueue)
        st_direction.emplace_back(previous_direction[0], previous_direction[1]);
        previous_direction[0] = vert[0] - mid_ctrl[0];
        previous_direction[1] = vert[1] - mid_ctrl[1];
        normalize(previous_direction[0], previous_direction[1]);
        end_direction.emplace_back(previous_direction[0], previous_control[1]);
    }
    int front_count = 0;
    int back_count = primitives.size();
    int last = back_count - 1;
    double smooth_radius = 5. / resolution;
    if (back_count - front_count != 1) // Only 1 curve
    {
        double mid[2], ed[2];
        if (primitives.back() == 0)
        {
            normalize(end_direction[last].x, end_direction[last].y);
            double d_endx = end_direction[last].x * smooth_radius;
            double d_endy = end_direction[last].y * smooth_radius;
            mid[0] = parameters[last].param[0];
            mid[1] = parameters[last].param[1];
            parameters[last].param[0] -= d_endx;
            parameters[last].param[1] -= d_endy;
        }
        else
        {
            normalize(end_direction[last].x, end_direction[last].y);
            double d_endx = end_direction[last].x * smooth_radius;
            double d_endy = end_direction[last].y * smooth_radius;
            mid[0] = parameters[last].param[2];
            mid[1] = parameters[last].param[3];
            parameters[last].param[2] -= d_endx;
            parameters[last].param[3] -= d_endy;
        }
        double d_stx = st_direction[0].x * smooth_radius;
        double d_sty = st_direction[0].y * smooth_radius;
        ed[0] = (vertices_data + 0)[0] + d_stx;
        ed[1] = (vertices_data + 0)[1] + d_sty;
        primitives.push_back(1);
        parameters.emplace_back(mid[0], mid[1], ed[0], ed[1]);
        back_count++;
    }

    py::array_t<int> numpy_primitives = py::array_t<int>(back_count-front_count);
    py::array_t<double> numpy_parameters = py::array_t<double>((back_count-front_count)*4);
    int* primitives_ptr = (int*)(numpy_primitives.request().ptr);
    double* parameters_ptr = (double*)(numpy_parameters.request().ptr);
    // for (int i = 0; i < primitives.size(); i++)
    // {
    //     std::cout << primitives[i] << " ";
    // }
    // std::cout << std::endl;
    for (int i = front_count; i < back_count; i++)
    {
        int offset_i = i - front_count;
        primitives_ptr[offset_i] = primitives[i];
        parameters_ptr[offset_i*4] = parameters[i].param[0];
        parameters_ptr[offset_i*4+1] = parameters[i].param[1];
        parameters_ptr[offset_i*4+2] = parameters[i].param[2];
        parameters_ptr[offset_i*4+3] = parameters[i].param[3];
    }

    py::tuple result = py::tuple(2);
    result[0] = numpy_primitives;
    result[1] = numpy_parameters;
    return result;
}

auto Vectorize_Edge(
    py::array_t<double>& ordered_vertices, 
    py::array_t<double>& corner_st, 
    py::array_t<double>& corner_end,
    double resolution = 1024,
    double err_L_coe = 2, 
    double err_Q_coe = 2.)
{
    
    py::buffer_info vertices_buffer = ordered_vertices.request();
    py::buffer_info cst_buffer = corner_st.request();
    py::buffer_info cend_buffer = corner_end.request();
    if (vertices_buffer.ndim != 2 || vertices_buffer.shape[1] != 2)
    {
        throw std::runtime_error("Number of dimensions of arg0 [ordered_vertices] must be 2");
    }
    ssize_t dim0 = vertices_buffer.shape[0], dim1 = vertices_buffer.shape[1];
    double* vertices_data = (double*)vertices_buffer.ptr;
    double* cst_data = (double*)cst_buffer.ptr;
    double* cend_data = (double*)cend_buffer.ptr;
    // Hyper parameters
    const double err_threshold_L_ave = 1. / resolution;
    const double err_threshold_L_max = err_L_coe / resolution;
    const double err_threshold_Q_ave = 1. / resolution;
    const double err_threshold_Q_max = err_Q_coe / resolution;
    const double save_L_minlen = 0.1;
    const double num_min_init = 10;
    const double delta_diff = 1. / resolution;
    const double bisection_min_interval = 1. / resolution;
    // const double err_threshold_inc = 1. / 1024;
    const double PI = 3.14159265358979;
    const double smooth_min_angle = PI * 5. / 180.;
    bool init_flag = true;
    ssize_t lenv = vertices_buffer.shape[0];
    
    std::vector<double*> vqueue; // Read Only Reference to ordered_vertices
    std::vector<int> primitives;
    std::vector<param4> parameters;
    std::vector<Coords> prim_verts;
    std::vector<Coords> st_direction;
    std::vector<Coords> end_direction;
    std::vector<double*> controls;
    double* previous_control = vertices_data + 0;
    double previous_direction[2] = {vertices_data[1*dim0+0]-vertices_data[0*dim0+0],vertices_data[1*dim0+1]-vertices_data[0*dim0+1]};
    normalize(previous_direction[0], previous_direction[1]);

    double l_bisection = 0, r_bisection = 0.1;
    int temp_prim = -1;
    int vert_i = 0;
    double* vert;
    double mid_ctrl[2] {}, dif_mid_ctrl[2] {};
    while (vert_i < lenv)
    {
        // std::cout << vert_i << std::endl;
        vert = vertices_data + vert_i*dim1;
        
        if (vqueue.size() == 0)
        {
            vqueue.push_back(vert);
            controls.push_back(vert);
            previous_control = vert;
            // std::cout << "Empty vqueue\n";
        }
        else if (vqueue.size() < num_min_init)
        {
            // vert_i++;
            vqueue.push_back(vert);
            // pass
        }
        else if (temp_prim == -1)
        {
            vqueue.push_back(vert);
            temp_prim = 0;
            l_bisection = 0;
            r_bisection = 0.1;
        }
        else if (temp_prim == 0)
        {
            vqueue.push_back(vert);
            // std::cout << "prim 0" << std::endl;
            double x0 = previous_control[0], y0 = previous_control[1];
            double x1 = vert[0], y1 = vert[1];
            double a = y1 - y0;
            double b = x0 - x1;
            double c = x1*y0 - x0*y1;
            double proj = sqrt(a*a+b*b);
            double accumulate_err = 0.;
            double max_err = 0.;
            for (auto v : vqueue)
            {
                double err = abs(a*v[0] + b*v[1] + c) / proj;
                accumulate_err += err;
                max_err = max_err > err ? max_err : err;
            }
            double temp_direction[2];
            temp_direction[0] = vqueue.back()[0] - vqueue.front()[0];
            temp_direction[1] = vqueue.back()[1] - vqueue.front()[1];
            normalize(temp_direction[0], temp_direction[1]);
            double diff_angle = acos(dot(temp_direction, previous_direction));
            if ((diff_angle < smooth_min_angle) && (init_flag == false))
            {
                vert_i -= 1;
                temp_prim = 1;
                init_flag = false;
            }
            else 
            if ((accumulate_err / vqueue.size() > err_threshold_L_ave) || (max_err > err_threshold_L_max))
            {
                double v_p[2];
                v_p[0] = vert[0] - previous_control[0];
                v_p[1] = vert[1] - previous_control[1];
                if (norm(v_p) > save_L_minlen || init_flag == true)
                {
                    // std::cout << v_p[0] << " " << v_p[1] << std::endl;
                    // std::cout << "SAVE_PRIM_0\n";
                    vqueue.pop_back();
                    primitives.push_back(0);
                    // prim_verts.append(vqueue)
                    parameters.emplace_back(vqueue.back()[0], vqueue.back()[1]);
                    vqueue.clear();
                    vert_i -= 1;
                    temp_prim = -1;
                    previous_direction[0] = x1 - x0;
                    previous_direction[1] = y1 - y0;
                    normalize(previous_direction[0], previous_direction[1]);
                    st_direction.emplace_back(previous_direction[0], previous_direction[1]);
                    end_direction.emplace_back(previous_direction[0], previous_direction[1]);
                    init_flag = false;
                }
                else
                {
                    vert_i -= 1;
                    temp_prim = 1;
                    init_flag = false;
                }
                        
            }
        }
        else if (temp_prim == 1)
        {
            vqueue.push_back(vert);
            // std::cout << "prim 1" << std::endl;
            double* sample_t = linspace(vqueue.size() * 2);
            double* bezierM = Qbezier(sample_t, vqueue.size() * 2);
            // std::cout << "BEZM" << std::endl;
            double* errs = new double[vqueue.size()];
            // std::cout << "ALLOC1" << std::endl;
            double* dif_errs = new double[vqueue.size()];
            // std::cout << "ALLOC2" << std::endl;
            double* bez_points = new double[vqueue.size()*2*2];
            // std::cout << "ALLOC3" << std::endl;
            double* dif_bez_points = new double[vqueue.size()*2*2];
            // std::cout << "ALLOC4" << std::endl;
            for (int i = 0; i < vqueue.size(); i++)
            {
                errs[i] = 0;
                dif_errs[i] = 0;
            }
            while(true)
            {
                double lctrl[2], temp[2];
                lctrl[0] = previous_control[0] + previous_direction[0] * l_bisection;
                lctrl[1] = previous_control[1] + previous_direction[1] * l_bisection;
                temp[0] = vert[0] - lctrl[0];
                temp[1] = vert[1] - lctrl[1];
                double len_l2v = sqrt(temp[0]*temp[0] + temp[1]*temp[1]);
                if (l_bisection / len_l2v < 0.1)
                {
                    l_bisection += bisection_min_interval * 2;
                }
                else if (r_bisection / len_l2v > 0.9)
                {
                    r_bisection -= bisection_min_interval * 2;
                }
                else
                {
                    break;
                }
            }
            if (r_bisection < l_bisection)
            {
                r_bisection = l_bisection + bisection_min_interval * 8;
            }
            while(r_bisection - l_bisection > bisection_min_interval)
            {
                // break;
                // std::cout << l_bisection << " " << r_bisection << std::endl;
                double mid_bisection = (l_bisection + r_bisection) / 2;
                mid_ctrl[0] = previous_control[0] + previous_direction[0] * mid_bisection;
                mid_ctrl[1] = previous_control[1] + previous_direction[1] * mid_bisection;
                dif_mid_ctrl[0] = previous_control[0] + previous_direction[0] * (mid_bisection + delta_diff);
                dif_mid_ctrl[1] = previous_control[1] + previous_direction[1] * (mid_bisection + delta_diff);
                double ctrls[6], dif_ctrls[6];
                ctrls[0] = previous_control[0];
                ctrls[1] = previous_control[1];
                ctrls[2] = mid_ctrl[0];
                ctrls[3] = mid_ctrl[1];
                ctrls[4] = vert[0];
                ctrls[5] = vert[1];
                dif_ctrls[0] = previous_control[0];
                dif_ctrls[1] = previous_control[1];
                dif_ctrls[2] = dif_mid_ctrl[0];
                dif_ctrls[3] = dif_mid_ctrl[1];
                dif_ctrls[4] = vert[0];
                dif_ctrls[5] = vert[1];
                
                matmul_Qbezier(bezierM, vqueue.size() * 2, ctrls, bez_points);
                matmul_Qbezier(bezierM, vqueue.size() * 2, dif_ctrls, dif_bez_points);

                // printf("QB-----");
                chamfer_bezier(vqueue, bez_points, vqueue.size() * 2, errs);
                // printf("QBDF-----");
                chamfer_bezier(vqueue, dif_bez_points, vqueue.size() * 2, dif_errs);

                // for (int i = 0; i < vqueue.size()*2; i++)
                // {
                //     // printf("%lf ", bezierM[i]);
                //     printf("%lf %lf -", (bez_points[i*2]), (dif_bez_points[i*2]));
                // }

                double maxdiferr = max_err(dif_errs, vqueue.size());
                double maxerr = max_err(errs, vqueue.size());
                // std::cout << maxdiferr << " " << maxerr << std::endl;
                if (maxdiferr < maxerr)
                {
                    l_bisection = mid_bisection;
                }
                else
                {
                    r_bisection = mid_bisection;
                }
                // printf("%lf %lf %lf %lf-", ctrls[2], ctrls[3], dif_ctrls[2], dif_ctrls[3]);
                // printf("%lf %lf-", maxerr, maxdiferr);
            }
            // for (int i = 0; i < vqueue.size()*2; i++)
            // {
                // printf("%lf ", bezierM[i]);
                // printf("%lf %lf -", (bez_points[i*2]-dif_bez_points[i*2]), (bez_points[i*2+1]-dif_bez_points[i*2+1]));
            // }
            // for (int i = 0; i < vqueue.size(); i++)
            // {
            //     printf("%.9lf-", errs[i] - dif_errs[i]);
            // }
            // std::cout << "end bisection" << std::endl;
            l_bisection -= bisection_min_interval * 16;
            r_bisection += bisection_min_interval * 16;
            if (l_bisection < 0)
            {
                l_bisection = 0;
            }
            double avg_err = mean_err(errs, vqueue.size());
            double m_err = max_err(errs, vqueue.size());
            // printf("%lf\n", m_err);
            // if ((avg_err > err_threshold_Q_ave) ||(m_err > err_threshold_Q_max))
            if (m_err > err_threshold_Q_max)
            {
                // printf("(%lf %lf)\n", m_err, err_threshold_Q_max);
                // std::cout << "CLEAR" << std::endl;
                vqueue.pop_back();
                primitives.push_back(1);
                parameters.emplace_back(dif_mid_ctrl[0], dif_mid_ctrl[1], vert[0], vert[1]);
                // prim_verts.append(vqueue)
                vqueue.clear();
                // # Reset
                vert_i -= 1;
                temp_prim = -1;
                st_direction.emplace_back(previous_direction[0], previous_direction[1]);
                previous_direction[0] = vert[0] - mid_ctrl[0];
                previous_direction[1] = vert[1] - mid_ctrl[1];
                normalize(previous_direction[0], previous_direction[1]);
                end_direction.emplace_back(previous_direction[0], previous_direction[1]);
            }
            // std::cout << "Free memory\n";
            delete[] sample_t;
            delete[] bezierM;
            delete[] errs;
            delete[] dif_errs;
            delete[] bez_points;
            delete[] dif_bez_points;
            
        }
        vert_i += 1;
    }

    // std::cout << "Complete vertices\n";
    // std::cout << primitives.size() << std::endl;

    if (temp_prim == 0)
    {
        primitives.push_back(0);
        parameters.emplace_back(vqueue.back()[0], vqueue.back()[1]);
        // prim_verts.append(vqueue)
        previous_direction[0] = vqueue.back()[0] - vqueue.front()[0];
        previous_direction[1] = vqueue.back()[1] - vqueue.front()[1];
        normalize(previous_direction[0], previous_direction[1]);
        st_direction.emplace_back(previous_direction[0], previous_direction[1]);
        end_direction.emplace_back(previous_direction[0], previous_direction[1]);
    }
    else if (temp_prim == 1)
    {
        double* sample_t = linspace(vqueue.size() * 2);
            double* bezierM = Qbezier(sample_t, vqueue.size() * 2);
            double* errs = new double[vqueue.size()];
            double* dif_errs = new double[vqueue.size()];
            double* bez_points = new double[vqueue.size()*2*2];
            double* dif_bez_points = new double[vqueue.size()*2*2];
        while(r_bisection - l_bisection > bisection_min_interval)
            {
                // break;
                // std::cout << l_bisection << " " << r_bisection << std::endl;
                double mid_bisection = (l_bisection + r_bisection) / 2;
                mid_ctrl[0] = previous_control[0] + previous_direction[0] * mid_bisection;
                mid_ctrl[1] = previous_control[1] + previous_direction[1] * mid_bisection;
                dif_mid_ctrl[0] = previous_control[0] + previous_direction[0] * (mid_bisection + delta_diff);
                dif_mid_ctrl[1] = previous_control[1] + previous_direction[1] * (mid_bisection + delta_diff);
                double ctrls[6], dif_ctrls[6];
                ctrls[0] = previous_control[0];
                ctrls[1] = previous_control[1];
                ctrls[2] = mid_ctrl[0];
                ctrls[3] = mid_ctrl[1];
                ctrls[4] = vert[0];
                ctrls[5] = vert[1];
                dif_ctrls[0] = previous_control[0];
                dif_ctrls[1] = previous_control[1];
                dif_ctrls[2] = dif_mid_ctrl[0];
                dif_ctrls[3] = dif_mid_ctrl[1];
                dif_ctrls[4] = vert[0];
                dif_ctrls[5] = vert[1];
                matmul_Qbezier(bezierM, vqueue.size() * 2, ctrls, bez_points);
                matmul_Qbezier(bezierM, vqueue.size() * 2, dif_ctrls, dif_bez_points);
                
                chamfer_bezier(vqueue, bez_points, vqueue.size() * 2, errs);
                chamfer_bezier(vqueue, dif_bez_points, vqueue.size() * 2, dif_errs);

                double meandiferr = mean_err(dif_errs, vqueue.size());
                double meanerr = mean_err(errs, vqueue.size());
                // std::cout << meandiferr << " " << meanerr << std::endl;
                if (meandiferr < meanerr)
                {
                    l_bisection = mid_bisection;
                }
                else
                {
                    r_bisection = mid_bisection;
                }
            }
        delete[] sample_t;
            delete[] bezierM;
            delete[] errs;
            delete[] dif_errs;
            delete[] bez_points;
            delete[] dif_bez_points;
        primitives.push_back(1);
        parameters.emplace_back(dif_mid_ctrl[0], dif_mid_ctrl[1], vert[0], vert[1]);
        // prim_verts.append(vqueue)
        st_direction.emplace_back(previous_direction[0], previous_direction[1]);
        previous_direction[0] = vert[0] - mid_ctrl[0];
        previous_direction[1] = vert[1] - mid_ctrl[1];
        normalize(previous_direction[0], previous_direction[1]);
        end_direction.emplace_back(previous_direction[0], previous_control[1]);
    }
    int front_count = 0;
    int back_count = primitives.size();

    // std::cout << "Complete remains\n";



    
    for (size_t i = 0; i < primitives.size(); i++)
    {
        int prim = primitives[i];
        if (prim == 0)
        {
            // std::cout << "front 0" << std::endl;
            double temp_direction[2];
            temp_direction[0] = parameters[i].param[0] - cst_data[0];
            temp_direction[1] = parameters[i].param[1] - cst_data[1];
            normalize(temp_direction[0], temp_direction[1]);
            double diff_angle = acos(temp_direction[0]*st_direction[i].x+temp_direction[1]*st_direction[i].y);
            double __Temp_var[2];
            __Temp_var[0] = parameters[i].param[0] - cst_data[0];
            __Temp_var[1] = parameters[i].param[1] - cst_data[1];
            if ((diff_angle < smooth_min_angle) || norm(__Temp_var) > 0.05)
            {
                controls[i] = cst_data;
                break;
            }
        }
        else if (prim == 1)
        {
            // std::cout << "front 1" << std::endl;
            // printf("%lf %lf\n", controls[i][0], controls[i][1]);
            double __Temp_var[2];
            __Temp_var[0] = parameters[i].param[0] - cst_data[0];
            __Temp_var[1] = parameters[i].param[1] - cst_data[1];
            double cross1, cross2;
            cross1 = st_direction[i].x*end_direction[i].y - st_direction[i].y*end_direction[i].x;
            cross2 = __Temp_var[0]*end_direction[i].y - __Temp_var[1]*end_direction[i].x;
            
            if (norm(__Temp_var) > 0.05 || cross1 * cross2 > 0)
            {
                controls[i] = cst_data;
                break;
            }
        }
        front_count += 1;
    }
    
    // std::cout << "Complete corner_st\n";

    // rev_prims = reversed(list(range(front_count, back_count)))
    // printf("---\n");
    for (int i = back_count-1; i >= front_count; i--)
    {
        int prim = primitives[i];
        if (primitives[i] == 0)
        {
            // std::cout << "back 0" << std::endl;
            double temp_direction[2];
            temp_direction[0] = cend_data[0] - controls[i][0];
            temp_direction[1] = cend_data[1] - controls[i][1];
            double len_dir = norm(temp_direction);
            normalize(temp_direction[0], temp_direction[1]);
            double diff_angle = acos(temp_direction[0]*end_direction[i].x+temp_direction[1]*end_direction[i].y);
            if (len_dir > 0.1 || diff_angle < smooth_min_angle || back_count == 1)
            {
                // if (len_dir > 0.05)
                // {
                //     printf("ldd %lf, %lf\n", controls[i][0]*512, controls[i][1]*512);
                //     printf("-- %lf, %lf, %lf\n", cend_data[0]*512, cend_data[1]*512, len_dir);
                //     printf("-- %lf %lf\n", )
                // }
                // if (diff_angle < smooth_min_angle)
                //     printf("ds\n");
                // if (back_count == 1)
                //     printf("bc\n");
                parameters[i].param[0] = cend_data[0];
                parameters[i].param[1] = cend_data[1];
                break;
            }
        }
        else if (prim == 1)
        {
            // std::cout << "back 1" << std::endl;
            double __Temp_var[2];
            __Temp_var[0] = parameters[i].param[0] - cend_data[0];
            __Temp_var[1] = parameters[i].param[1] - cend_data[1];
            if (norm(__Temp_var) > 0.05)
            {
                parameters[i].param[2] = cend_data[0];
                parameters[i].param[3] = cend_data[1];
                break;
            }
        }
        back_count -= 1;
    }

    // std::cout << "Complete corner_end\n";
    // std::cout << front_count << " " << back_count << std::endl;
    if (front_count >= back_count)
    {
        // std::cout << "f > b" << std::endl;
        py::array_t<int> numpy_primitives = py::array_t<int>(1);
        py::array_t<double> numpy_parameters = py::array_t<double>(1*4);
        double* primitives_ptr = (double*)(numpy_primitives.request().ptr);
        double* parameters_ptr = (double*)(numpy_parameters.request().ptr);
        primitives_ptr[0] = 0;
        parameters_ptr[0] = cend_data[0];
        parameters_ptr[1] = cend_data[1];
        parameters_ptr[2] = 0;
        parameters_ptr[3] = 0;
        // std::cout << "require tuple" << std::endl;
        py::tuple result = py::tuple(2);
        result[0] = numpy_primitives;
        result[1] = numpy_parameters;
        return result;
    }

    // std::cout << primitives.size() << " " << parameters.size() << std::endl;
    py::array_t<int> numpy_primitives = py::array_t<int>(back_count-front_count);
    py::array_t<double> numpy_parameters = py::array_t<double>((back_count-front_count)*4);
    int* primitives_ptr = (int*)(numpy_primitives.request().ptr);
    double* parameters_ptr = (double*)(numpy_parameters.request().ptr);
    // for (int i = 0; i < primitives.size(); i++)
    // {
    //     std::cout << primitives[i] << " ";
    // }
    // std::cout << std::endl;
    for (int i = front_count; i < back_count; i++)
    {
        int offset_i = i - front_count;
        primitives_ptr[offset_i] = primitives[i];
        parameters_ptr[offset_i*4] = parameters[i].param[0];
        parameters_ptr[offset_i*4+1] = parameters[i].param[1];
        parameters_ptr[offset_i*4+2] = parameters[i].param[2];
        parameters_ptr[offset_i*4+3] = parameters[i].param[3];
    }

    py::tuple result = py::tuple(2);
    result[0] = numpy_primitives;
    result[1] = numpy_parameters;
    
    return result;
}

double* chamfer_bezier(std::vector<double*>& vqueue, double* bez_points, ssize_t n_bezier, double* errs)
{
    int count_set2 = 0;
    for (size_t i = 0; i < vqueue.size(); i++)
    {
        double* p = vqueue[i];
        
        double* tempcmpp = bez_points + count_set2*2;
        double err_[2];
        err_[0] = tempcmpp[0] - p[0];
        err_[1] = tempcmpp[1] - p[1];
        double err = sqrt(err_[0]*err_[0]+err_[1]*err_[1]);
        count_set2++;
        while (count_set2 < n_bezier)
        {
            // std::cout << count_set2 << " ";
            tempcmpp = bez_points + count_set2*2;
            double err_[2];
            err_[0] = tempcmpp[0] - p[0];
            err_[1] = tempcmpp[1] - p[1];
            double temperr = sqrt(err_[0]*err_[0]+err_[1]*err_[1]);
            // printf("%lfin", temperr);
            if (temperr < err)
            {
                err = temperr;
                count_set2 += 1;
            }
            else
            {
                break;
            }
        }
        if (count_set2 >= n_bezier)
            count_set2 -= 1;
        errs[i] = err;
        // std::cout << err << " ";
    }
    // std::cout << std::endl;
    return errs;
}

PYBIND11_MODULE(vct, m) {
    m.doc() = "pybind11 example plugin"; // optional module docstring

    m.def("check_mask", &check_mask, "A function which adds two numbers");

    m.def("check_pngs", &check_pngs, "");

    m.def("vectorize_edge", &Vectorize_Edge, " ");

    m.def("vectorize_wc", &Vectorize_wo_Corner, " ");
}