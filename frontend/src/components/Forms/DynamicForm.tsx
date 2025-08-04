import React from 'react';
import { Form, Button, Space, Card, Divider } from 'antd';
import type { FormProps } from 'antd';
import { FormField } from './FormField';
import type { FieldType } from './FormField';
import type { Rule } from 'antd/es/form';

interface FormFieldConfig {
  type: FieldType;
  name: string;
  label: string;
  placeholder?: string;
  options?: Array<{ label: string; value: any; disabled?: boolean }>;
  min?: number;
  max?: number;
  step?: number;
  rows?: number;
  disabled?: boolean;
  required?: boolean;
  rules?: Rule[];
  tooltip?: string;
  extra?: string;
  span?: number; // Grid span (1-24)
  dependencies?: string[]; // Field dependencies
  condition?: (values: any) => boolean; // Conditional rendering
}

interface FormSection {
  title?: string;
  description?: string;
  fields: FormFieldConfig[];
  collapsible?: boolean;
  defaultCollapsed?: boolean;
}

interface DynamicFormProps extends Omit<FormProps, 'onFinish'> {
  sections: FormSection[];
  onSubmit: (values: any) => void;
  onCancel?: () => void;
  loading?: boolean;
  submitText?: string;
  cancelText?: string;
  showCancel?: boolean;
  resetable?: boolean;
  resetText?: string;
  className?: string;
}

export const DynamicForm: React.FC<DynamicFormProps> = ({
  sections,
  onSubmit,
  onCancel,
  loading = false,
  submitText = '提交',
  cancelText = '取消',
  showCancel = false,
  resetable = false,
  resetText = '重置',
  className,
  ...formProps
}) => {
  const [form] = Form.useForm();

  const handleFinish = (values: any) => {
    onSubmit(values);
  };

  const handleReset = () => {
    form.resetFields();
  };

  const renderField = (field: FormFieldConfig, formValues: any) => {
    // Check conditional rendering
    if (field.condition && !field.condition(formValues)) {
      return null;
    }

    return (
      <div
        key={field.name}
        style={{
          gridColumn: field.span ? `span ${field.span}` : 'span 12',
        }}
      >
        <FormField
          type={field.type}
          name={field.name}
          label={field.label}
          placeholder={field.placeholder}
          options={field.options}
          min={field.min}
          max={field.max}
          step={field.step}
          rows={field.rows}
          disabled={field.disabled}
          required={field.required}
          rules={field.rules}
          tooltip={field.tooltip}
          extra={field.extra}
          dependencies={field.dependencies}
        />
      </div>
    );
  };

  const renderSection = (section: FormSection, index: number) => {
    return (
      <div key={index}>
        {section.title && (
          <>
            <Divider orientation="left">{section.title}</Divider>
            {section.description && (
              <p style={{ color: '#666', marginBottom: 16 }}>
                {section.description}
              </p>
            )}
          </>
        )}
        
        <Form.Item dependencies={[]}>
          {({ getFieldsValue }) => {
            const formValues = getFieldsValue();
            return (
              <div
                style={{
                  display: 'grid',
                  gridTemplateColumns: 'repeat(24, 1fr)',
                  gap: '16px',
                }}
              >
                {section.fields.map((field) => renderField(field, formValues))}
              </div>
            );
          }}
        </Form.Item>
      </div>
    );
  };

  return (
    <Card className={className}>
      <Form
        form={form}
        layout="vertical"
        onFinish={handleFinish}
        scrollToFirstError
        {...formProps}
      >
        {sections.map((section, index) => renderSection(section, index))}

        <Form.Item style={{ marginTop: 24 }}>
          <Space>
            <Button
              type="primary"
              htmlType="submit"
              loading={loading}
            >
              {submitText}
            </Button>
            
            {showCancel && (
              <Button onClick={onCancel}>
                {cancelText}
              </Button>
            )}
            
            {resetable && (
              <Button onClick={handleReset}>
                {resetText}
              </Button>
            )}
          </Space>
        </Form.Item>
      </Form>
    </Card>
  );
};

export default DynamicForm;