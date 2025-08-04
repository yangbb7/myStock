import React from 'react';
import {
  Form,
  Input,
  InputNumber,
  Select,
  DatePicker,
  Switch,
  Slider,
  Rate,
  Checkbox,
  Radio,
  Upload,
  Button,
} from 'antd';
import type { FormItemProps } from 'antd';
import { UploadOutlined } from '@ant-design/icons';
import type { Rule } from 'antd/es/form';

const { TextArea } = Input;
const { Option } = Select;
const { RangePicker } = DatePicker;

export type FieldType = 
  | 'input'
  | 'textarea'
  | 'number'
  | 'select'
  | 'multiselect'
  | 'date'
  | 'daterange'
  | 'switch'
  | 'slider'
  | 'rate'
  | 'checkbox'
  | 'radio'
  | 'upload';

interface Option {
  label: string;
  value: any;
  disabled?: boolean;
}

interface FormFieldProps extends Omit<FormItemProps, 'children'> {
  type: FieldType;
  name: string;
  label: string;
  placeholder?: string;
  options?: Option[];
  min?: number;
  max?: number;
  step?: number;
  rows?: number;
  disabled?: boolean;
  required?: boolean;
  rules?: Rule[];
  tooltip?: string;
  extra?: string;
  prefix?: React.ReactNode;
  suffix?: React.ReactNode;
  addonBefore?: React.ReactNode;
  addonAfter?: React.ReactNode;
  allowClear?: boolean;
  showSearch?: boolean;
  multiple?: boolean;
  accept?: string;
  maxCount?: number;
  onChange?: (value: any) => void;
  onBlur?: () => void;
  onFocus?: () => void;
}

export const FormField: React.FC<FormFieldProps> = ({
  type,
  name,
  label,
  placeholder,
  options = [],
  min,
  max,
  step,
  rows = 4,
  disabled = false,
  required = false,
  rules = [],
  tooltip,
  extra,
  prefix,
  suffix,
  addonBefore,
  addonAfter,
  allowClear = true,
  showSearch = false,
  multiple = false,
  accept,
  maxCount = 1,
  onChange,
  onBlur,
  onFocus,
  ...formItemProps
}) => {
  // Build validation rules
  const validationRules: Rule[] = [
    ...(required ? [{ required: true, message: `请输入${label}` }] : []),
    ...rules,
  ];

  const renderField = () => {
    const commonProps = {
      placeholder,
      disabled,
      onChange,
      onBlur,
      onFocus,
    };

    switch (type) {
      case 'input':
        return (
          <Input
            {...commonProps}
            prefix={prefix}
            suffix={suffix}
            addonBefore={addonBefore}
            addonAfter={addonAfter}
            allowClear={allowClear}
          />
        );

      case 'textarea':
        return (
          <TextArea
            {...commonProps}
            rows={rows}
            allowClear={allowClear}
          />
        );

      case 'number':
        return (
          <InputNumber
            {...commonProps}
            min={min}
            max={max}
            step={step}
            style={{ width: '100%' }}
            prefix={prefix}
            addonBefore={addonBefore}
            addonAfter={addonAfter}
          />
        );

      case 'select':
      case 'multiselect':
        return (
          <Select
            {...commonProps}
            mode={type === 'multiselect' ? 'multiple' : undefined}
            showSearch={showSearch}
            allowClear={allowClear}
            filterOption={(input, option) =>
              String(option?.label ?? '').toLowerCase().includes(input.toLowerCase())
            }
          >
            {options.map((option) => (
              <Option
                key={option.value}
                value={option.value}
                disabled={option.disabled}
              >
                {option.label}
              </Option>
            ))}
          </Select>
        );

      case 'date':
        return (
          <DatePicker
            {...commonProps}
            style={{ width: '100%' }}
            allowClear={allowClear}
          />
        );

      case 'daterange':
        return (
          <RangePicker
            {...commonProps}
            style={{ width: '100%' }}
            allowClear={allowClear}
            placeholder={['开始日期', '结束日期']}
          />
        );

      case 'switch':
        return <Switch {...commonProps} />;

      case 'slider':
        return (
          <Slider
            {...commonProps}
            min={min}
            max={max}
            step={step}
          />
        );

      case 'rate':
        return <Rate {...commonProps} />;

      case 'checkbox':
        return (
          <Checkbox.Group {...commonProps}>
            {options.map((option) => (
              <Checkbox
                key={option.value}
                value={option.value}
                disabled={option.disabled}
              >
                {option.label}
              </Checkbox>
            ))}
          </Checkbox.Group>
        );

      case 'radio':
        return (
          <Radio.Group {...commonProps}>
            {options.map((option) => (
              <Radio
                key={option.value}
                value={option.value}
                disabled={option.disabled}
              >
                {option.label}
              </Radio>
            ))}
          </Radio.Group>
        );

      case 'upload':
        return (
          <Upload
            {...commonProps}
            accept={accept}
            maxCount={maxCount}
            multiple={multiple}
          >
            <Button icon={<UploadOutlined />}>点击上传</Button>
          </Upload>
        );

      default:
        return <Input {...commonProps} />;
    }
  };

  return (
    <Form.Item
      name={name}
      label={label}
      rules={validationRules}
      tooltip={tooltip}
      extra={extra}
      {...formItemProps}
    >
      {renderField()}
    </Form.Item>
  );
};

export default FormField;